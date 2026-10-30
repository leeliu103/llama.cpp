// RDMA transport over ibverbs (rdma-core): one reliable-connected queue pair per peer; a message is an RDMA write
// with immediate into the peer's receive slot for this rank, completions polled (no interrupts). Queue pair numbers,
// GIDs and buffer keys go over the TCP transport on the same peer list. Included by transport.h with -DMK_VERBS;
// link with -libverbs. MK_IB_DEV picks the device (default the first), MK_IB_GID the GID index (default the first
// IPv4-mapped one, i.e. RoCE v2).
//
// Slots are double-buffered by message parity: a rank writes message k + 2 into the slot of message k only after
// it has received the peer's message k + 1, which the peer sends after it has read message k.
#pragma once

#include <infiniband/verbs.h>

struct VerbsTransport : Transport {
  static constexpr size_t kMax = 65536;  // bytes per message
  enum Kind { kColl = 0, kP2p = 1 };

  TcpTransport boot;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq *scq = nullptr, *rcq = nullptr;
  ibv_mr *rmr = nullptr, *smr = nullptr;
  char *rbuf = nullptr, *sbuf = nullptr;
  struct Peer {
    ibv_qp* qp = nullptr;
    uint64_t raddr = 0;
    uint32_t rkey = 0;
  };
  std::vector<Peer> peer;
  std::vector<unsigned> got[2];  // messages received from each peer, per kind
  std::vector<unsigned> p2p_sent, p2p_read;
  unsigned coll_seq = 0;
  int sends_out = 0;

  // receive slot of (sender, kind, parity) in a rank's receive buffer
  static size_t slot(int from, int kind, int parity) { return (((size_t)from * 2 + kind) * 2 + parity) * kMax; }

  VerbsTransport(int rank_, const std::vector<std::string>& hosts) : boot(rank_, hosts) {
    rank = boot.rank;
    size = boot.size;
    peer.resize(size);
    for (auto& g : got) g.assign(size, 0);
    p2p_sent.assign(size, 0);
    p2p_read.assign(size, 0);

    int ndev = 0;
    ibv_device** devs = ibv_get_device_list(&ndev);
    if (!devs || ndev == 0) die("no RDMA device");
    ibv_device* dev = devs[0];
    if (const char* want = getenv("MK_IB_DEV")) {
      dev = nullptr;
      for (int i = 0; i < ndev; i++)
        if (!strcmp(ibv_get_device_name(devs[i]), want)) dev = devs[i];
      if (!dev) die("MK_IB_DEV not found");
    }
    ctx = ibv_open_device(dev);
    ibv_free_device_list(devs);
    if (!ctx) die("ibv_open_device");
    ibv_port_attr port{};
    if (ibv_query_port(ctx, 1, &port)) die("ibv_query_port");
    int gid_index = -1;
    if (const char* g = getenv("MK_IB_GID")) gid_index = atoi(g);
    ibv_gid gid{};
    for (int i = 0; gid_index < 0 && i < port.gid_tbl_len; i++) {
      if (ibv_query_gid(ctx, 1, i, &gid)) continue;
      const uint8_t* b = gid.raw;
      bool v4 = b[10] == 0xff && b[11] == 0xff;
      for (int k = 0; k < 10; k++) v4 &= b[k] == 0;
      if (v4) gid_index = i;
    }
    if (gid_index < 0) gid_index = 0;
    if (ibv_query_gid(ctx, 1, gid_index, &gid)) die("ibv_query_gid");

    pd = ibv_alloc_pd(ctx);
    scq = ibv_create_cq(ctx, 256, nullptr, nullptr, 0);
    rcq = ibv_create_cq(ctx, 4096, nullptr, nullptr, 0);
    if (!pd || !scq || !rcq) die("pd / cq");
    const size_t rbytes = slot(size, 0, 0), sbytes = 4 * kMax;  // send: [kind][parity]
    rbuf = (char*)aligned_alloc(4096, rbytes);
    sbuf = (char*)aligned_alloc(4096, sbytes);
    memset(rbuf, 0, rbytes);
    memset(sbuf, 0, sbytes);
    rmr = ibv_reg_mr(pd, rbuf, rbytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    smr = ibv_reg_mr(pd, sbuf, sbytes, IBV_ACCESS_LOCAL_WRITE);
    if (!rmr || !smr) die("ibv_reg_mr (raise the memlock limit?)");

    struct Info {
      uint32_t qpn, psn, rkey, lid;
      uint64_t raddr;
      uint8_t gid[16];
    };
    std::vector<Info> mine(size), all((size_t)size * size);
    for (int p = 0; p < size; p++) {
      if (p == rank) continue;
      ibv_qp_init_attr qa{};
      qa.send_cq = scq;
      qa.recv_cq = rcq;
      qa.qp_type = IBV_QPT_RC;
      qa.cap.max_send_wr = 64;
      qa.cap.max_recv_wr = 512;
      qa.cap.max_send_sge = 1;
      qa.cap.max_recv_sge = 1;
      peer[p].qp = ibv_create_qp(pd, &qa);
      if (!peer[p].qp) die("ibv_create_qp");
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT;
      a.pkey_index = 0;
      a.port_num = 1;
      a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
      if (ibv_modify_qp(peer[p].qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        die("qp INIT");
      for (int i = 0; i < 512; i++) post_recv(p);
      mine[p] = {peer[p].qp->qp_num, (uint32_t)(rank * 1000 + p) & 0xffffff, rmr->rkey, port.lid,
                 (uint64_t)(uintptr_t)rbuf, {}};
      memcpy(mine[p].gid, gid.raw, 16);
    }
    boot.all_gather(mine.data(), all.data(), size * sizeof(Info));
    for (int p = 0; p < size; p++) {
      if (p == rank) continue;
      const Info& r = all[(size_t)p * size + rank];  // peer p's queue pair for me
      const Info& m = mine[p];
      peer[p].raddr = r.raddr;
      peer[p].rkey = r.rkey;
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR;
      a.path_mtu = port.active_mtu;
      a.dest_qp_num = r.qpn;
      a.rq_psn = r.psn;
      a.max_dest_rd_atomic = 1;
      a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1;
      memcpy(a.ah_attr.grh.dgid.raw, r.gid, 16);
      a.ah_attr.grh.sgid_index = gid_index;
      a.ah_attr.grh.hop_limit = 64;
      a.ah_attr.dlid = r.lid;
      a.ah_attr.port_num = 1;
      if (ibv_modify_qp(peer[p].qp, &a,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        die("qp RTR");
      ibv_qp_attr s{};
      s.qp_state = IBV_QPS_RTS;
      s.timeout = 14;
      s.retry_cnt = 7;
      s.rnr_retry = 7;
      s.sq_psn = m.psn;
      s.max_rd_atomic = 1;
      if (ibv_modify_qp(peer[p].qp, &s,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC))
        die("qp RTS");
    }
    boot.barrier();  // every receive is posted before anyone writes
  }
  ~VerbsTransport() override {
    for (auto& p : peer)
      if (p.qp) ibv_destroy_qp(p.qp);
    if (rmr) ibv_dereg_mr(rmr);
    if (smr) ibv_dereg_mr(smr);
    if (scq) ibv_destroy_cq(scq);
    if (rcq) ibv_destroy_cq(rcq);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    free(rbuf);
    free(sbuf);
  }
  const char* name() const override { return "verbs"; }

  [[noreturn]] static void die(const char* what) {
    fprintf(stderr, "verbs: %s\n", what);
    exit(1);
  }
  void post_recv(int p) {
    ibv_recv_wr wr{}, *bad;
    wr.wr_id = (uint64_t)p;
    if (ibv_post_recv(peer[p].qp, &wr, &bad)) die("ibv_post_recv");
  }
  void write(int p, int kind, int parity, size_t n) {
    ibv_sge sge{(uint64_t)(uintptr_t)(sbuf + ((size_t)kind * 2 + parity) * kMax), (uint32_t)n, smr->lkey};
    ibv_send_wr wr{}, *bad;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.imm_data = htonl((uint32_t)rank | (uint32_t)kind << 8);
    wr.wr.rdma.remote_addr = peer[p].raddr + slot(rank, kind, parity);
    wr.wr.rdma.rkey = peer[p].rkey;
    if (ibv_post_send(peer[p].qp, &wr, &bad)) die("ibv_post_send");
    sends_out++;
  }
  void poll() {
    ibv_wc wc[16];
    int n = ibv_poll_cq(rcq, 16, wc);
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) die(ibv_wc_status_str(wc[i].status));
      const uint32_t imm = ntohl(wc[i].imm_data);
      got[(imm >> 8) & 1][imm & 0xff]++;
      post_recv((int)wc[i].wr_id);
    }
    n = ibv_poll_cq(scq, 16, wc);
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) die(ibv_wc_status_str(wc[i].status));
      sends_out--;
    }
  }

  void all_gather(const void* mine, void* all, size_t n) override {
    if (n > kMax) die("message too large");
    const int parity = coll_seq & 1;
    memcpy(sbuf + (size_t)parity * kMax, mine, n);
    for (int p = 0; p < size; p++)
      if (p != rank) write(p, kColl, parity, n);
    for (;;) {
      bool done = sends_out == 0;
      for (int p = 0; p < size && done; p++) done = p == rank || got[kColl][p] > coll_seq;
      if (done) break;
      poll();
    }
    for (int p = 0; p < size; p++)
      memcpy((char*)all + (size_t)p * n, p == rank ? (const char*)mine : rbuf + slot(p, kColl, parity), n);
    coll_seq++;
  }
  void send(int p, const void* buf, size_t n) override {
    const int parity = p2p_sent[p]++ & 1;
    memcpy(sbuf + (2 + (size_t)parity) * kMax, buf, n);
    write(p, kP2p, parity, n);
    while (sends_out) poll();
  }
  void recv(int p, void* buf, size_t n) override {
    while (got[kP2p][p] <= p2p_read[p]) poll();
    const int parity = p2p_read[p]++ & 1;
    memcpy(buf, rbuf + slot(p, kP2p, parity), n);
  }
  size_t try_send(int, const void*, size_t) override { abort(); }
  size_t try_recv(int, void*, size_t) override { abort(); }
};
