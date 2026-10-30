// OdinLink transport: USB4 through the OdinLink driver (odl_tb5.ko), not the kernel network stack. Each cable is
// one /dev/odl_tb5_<n>, so each peer is one device; a message is one stream message on it (the driver cuts it into
// 4 KB frames and reassembles it; the receive ioctl sleeps until a whole message is there). Devices are matched to
// ranks by link UUID over the TCP transport on the same peer list, so a box needs a direct cable to every peer.
// Included by transport.h with -DMK_ODL; ioctls only, no library. MK_ODL_STREAM picks the stream id (default 91;
// a message to a stream the peer has not opened is dropped, so every rank opens it before the first send).
#pragma once

#include <sys/ioctl.h>

namespace odl {
// odl_tb5_uapi.h of the loaded driver (OdinLink-Five 8b8b84c)
struct PeerInfo {
  uint8_t uuid[16];
  uint32_t link_speed, link_width, state, reserved;
  char vendor_name[64], device_name[64];
};
struct LinkInfo {
  uint8_t local_uuid[16], peer_uuid[16];
  uint32_t link_speed, link_width, state, reserved;
  char vendor_name[64], device_name[64];
};
struct StreamReq {
  uint8_t stream_id, flags;
};
struct StreamXfer {
  uint8_t stream_id, dst_id, src_id, flags;
  uint64_t data;
  uint32_t len, actual_len;
};
constexpr unsigned long kGetPeer = _IOR('O', 0x07, PeerInfo);
constexpr unsigned long kGetLinkInfo = _IOR('O', 0x28, LinkInfo);
constexpr unsigned long kStreamOpen = _IOWR('O', 0x20, StreamReq);
constexpr unsigned long kStreamClose = _IOW('O', 0x21, StreamReq);
constexpr unsigned long kStreamSend = _IOW('O', 0x22, StreamXfer);
constexpr unsigned long kStreamRecv = _IOWR('O', 0x23, StreamXfer);
constexpr uint32_t kReady = 4;
constexpr int kMaxDevices = 16, kMaxLinks = 4;
}  // namespace odl

struct OdlTransport : Transport {
  TcpTransport boot;
  std::vector<int> fd;  // device of the cable to each peer rank
  uint8_t stream = 91;

  OdlTransport(int rank_, const std::vector<std::string>& hosts) : boot(rank_, hosts) {
    rank = boot.rank;
    size = boot.size;
    if (const char* s = getenv("MK_ODL_STREAM")) stream = (uint8_t)atoi(s);

    struct Links {
      uint32_t n;
      uint8_t local[odl::kMaxLinks][16];
    } mine{};
    std::vector<int> dev;
    std::vector<odl::LinkInfo> info;
    for (int i = 0; i < odl::kMaxDevices && (int)dev.size() < odl::kMaxLinks; i++) {
      const std::string path = "/dev/odl_tb5_" + std::to_string(i);
      const int f = open(path.c_str(), O_RDWR);
      if (f < 0) continue;
      odl::PeerInfo p{};
      const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      while (ioctl(f, odl::kGetPeer, &p) == 0 && p.state != odl::kReady &&
             std::chrono::steady_clock::now() < give_up)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      odl::LinkInfo li{};
      if (p.state != odl::kReady || ioctl(f, odl::kGetLinkInfo, &li) != 0) {
        fprintf(stderr, "odl: %s has no ready peer, skipped\n", path.c_str());
        close(f);
        continue;
      }
      memcpy(mine.local[mine.n++], li.local_uuid, 16);
      dev.push_back(f);
      info.push_back(li);
    }
    std::vector<Links> all(size);
    boot.all_gather(&mine, all.data(), sizeof(Links));
    fd.assign(size, -1);
    for (int p = 0; p < size; p++) {
      if (p == rank) continue;
      for (size_t d = 0; d < dev.size() && fd[p] < 0; d++)
        for (uint32_t k = 0; k < all[p].n; k++)
          if (!memcmp(info[d].peer_uuid, all[p].local[k], 16)) fd[p] = dev[d];
      if (fd[p] < 0) {
        fprintf(stderr, "odl: rank %d: no direct cable to rank %d\n", rank, p);
        exit(1);
      }
      odl::StreamReq req{stream, 0};
      if (ioctl(fd[p], odl::kStreamOpen, &req) != 0) xfer_die("stream open (MK_ODL_STREAM?)");
      if (req.stream_id != stream) {
        fprintf(stderr, "odl: asked for stream %u, got %u\n", stream, req.stream_id);
        exit(1);
      }
    }
    for (int d : dev)
      if (std::find(fd.begin(), fd.end(), d) == fd.end()) close(d);
    boot.barrier();  // every stream is open before anyone sends
  }
  ~OdlTransport() override {
    for (int p = 0; p < size; p++) {
      if (fd[p] < 0) continue;
      odl::StreamReq req{stream, 0};
      ioctl(fd[p], odl::kStreamClose, &req);
      close(fd[p]);
    }
  }
  const char* name() const override { return "odl"; }

  void post(int p, const void* buf, size_t n) {
    odl::StreamXfer x{stream, stream, 0, 0, (uint64_t)(uintptr_t)buf, (uint32_t)n, 0};
    while (ioctl(fd[p], odl::kStreamSend, &x) != 0)
      if (errno != EINTR) xfer_die("stream send");
  }
  // One message per call on each side, so a message is the n bytes asked for.
  void take(int p, void* buf, size_t n) {
    odl::StreamXfer x{stream, 0, 0, 0, (uint64_t)(uintptr_t)buf, (uint32_t)n, 0};
    while (ioctl(fd[p], odl::kStreamRecv, &x) != 0)
      if (errno != EINTR && errno != EAGAIN) xfer_die("stream recv");
    if (x.actual_len != n) {
      fprintf(stderr, "odl: rank %d: %u bytes from rank %d, expected %zu\n", rank, x.actual_len, p, n);
      exit(1);
    }
  }

  void all_gather(const void* mine, void* all, size_t n) override {
    for (int p = 0; p < size; p++)
      if (p != rank) post(p, mine, n);
    memcpy((char*)all + (size_t)rank * n, mine, n);
    for (int p = 0; p < size; p++)
      if (p != rank) take(p, (char*)all + (size_t)p * n, n);
  }
  void send(int p, const void* buf, size_t n) override { post(p, buf, n); }
  void recv(int p, void* buf, size_t n) override { take(p, buf, n); }
  size_t try_send(int, const void*, size_t) override { abort(); }
  size_t try_recv(int, void*, size_t) override { abort(); }
};
