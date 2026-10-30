// Small-message transport between the ranks of a tensor split, one rank per box. TCP for now; a
// USB4-RDMA backend implements try_send / try_recv the same way. Every rank runs the same sequence of
// operations, so each peer's byte stream stays aligned without headers.
#pragma once

#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

[[noreturn]] inline void xfer_die(const char* what) {
  fprintf(stderr, "transport: %s: %s\n", what, strerror(errno));
  exit(1);
}

struct Transport {
  int rank = 0, size = 1;
  virtual ~Transport() = default;
  virtual const char* name() const = 0;
  // Move what can be moved without blocking; return the bytes done (0 if none).
  virtual size_t try_send(int peer, const void* buf, size_t n) = 0;
  virtual size_t try_recv(int peer, void* buf, size_t n) = 0;

  virtual void send(int peer, const void* buf, size_t n) {
    for (size_t d = 0; d < n;) d += try_send(peer, (const char*)buf + d, n - d);
  }
  virtual void recv(int peer, void* buf, size_t n) {
    for (size_t d = 0; d < n;) d += try_recv(peer, (char*)buf + d, n - d);
  }
  // all[p * n] = rank p's n bytes. Progresses every peer at once, so large messages cannot deadlock
  // on full socket buffers.
  virtual void all_gather(const void* mine, void* all, size_t n) {
    char* a = (char*)all;
    memcpy(a + (size_t)rank * n, mine, n);
    std::vector<size_t> sent(size, n), got(size, n);
    for (int p = 0; p < size; p++)
      if (p != rank) sent[p] = got[p] = 0;
    for (int left = 2 * (size - 1); left > 0;) {
      for (int p = 0; p < size; p++) {
        if (sent[p] < n && (sent[p] += try_send(p, (const char*)mine + sent[p], n - sent[p])) == n) left--;
        if (got[p] < n && (got[p] += try_recv(p, a + (size_t)p * n + got[p], n - got[p])) == n) left--;
      }
    }
  }
  // Every rank ends with the same sum, added in rank order: bit-identical on all ranks.
  void allreduce_sum(float* x, size_t n, std::vector<float>& scratch) {
    if (size == 1) return;
    scratch.resize(n * size);
    all_gather(x, scratch.data(), n * 4);
    for (size_t i = 0; i < n; i++) {
      float s = scratch[i];
      for (int p = 1; p < size; p++) s += scratch[(size_t)p * n + i];
      x[i] = s;
    }
  }
  // The same with every rank's n floats rounded to fp16 on the wire (half the bytes), added in fp32 in rank order;
  // a rank's own partial is rounded too, so the sum stays bit-identical on all ranks. n a multiple of 8.
  void allreduce_sum16(float* x, size_t n, std::vector<uint16_t>& scratch) {
    if (size == 1) return;
    scratch.resize(n * (size + 1));
    uint16_t* mine = scratch.data() + n * size;
    f32_to_f16(x, mine, n);
    all_gather(mine, scratch.data(), n * 2);
    sum_f16(scratch.data(), size, n, x);
  }
  __attribute__((target("avx2,f16c"))) static void f32_to_f16(const float* x, uint16_t* h, size_t n) {
    for (size_t i = 0; i < n; i += 8)
      _mm_storeu_si128((__m128i*)(h + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
  }
  __attribute__((target("avx2,f16c"))) static void sum_f16(const uint16_t* all, int ranks, size_t n, float* x) {
    for (size_t i = 0; i < n; i += 8) {
      __m256 s = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(all + i)));
      for (int p = 1; p < ranks; p++)
        s = _mm256_add_ps(s, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(all + (size_t)p * n + i))));
      _mm256_storeu_ps(x + i, s);
    }
  }
  void barrier() {
    const char one = 1;
    std::vector<char> all(size);
    all_gather(&one, all.data(), 1);
  }
};

struct LocalTransport : Transport {
  const char* name() const override { return "local"; }
  size_t try_send(int, const void*, size_t) override { abort(); }
  size_t try_recv(int, void*, size_t) override { abort(); }
};

// The ranks of one process (shards emulated on one machine): every rank writes its bytes into its slot,
// meets the others, copies all slots, meets them again before the slots are reused.
struct Mesh {
  explicit Mesh(int n) : size(n), slot(n) {}
  int size;
  std::vector<std::vector<char>> slot;
  std::atomic<unsigned> count{0}, gen{0};
  void meet() {
    const unsigned g = gen.load(std::memory_order_acquire);
    if (count.fetch_add(1, std::memory_order_acq_rel) == (unsigned)size - 1) {
      count.store(0, std::memory_order_relaxed);
      gen.fetch_add(1, std::memory_order_release);
    } else {
      while (gen.load(std::memory_order_acquire) == g) {
      }
    }
  }
};

struct MeshTransport : Transport {
  Mesh* m;
  MeshTransport(Mesh* mesh, int r) : m(mesh) {
    rank = r;
    size = mesh->size;
  }
  const char* name() const override { return "mesh"; }
  size_t try_send(int, const void*, size_t) override { abort(); }
  size_t try_recv(int, void*, size_t) override { abort(); }
  void all_gather(const void* mine, void* all, size_t n) override {
    auto& s = m->slot[rank];
    if (s.size() < n) s.resize(n);
    memcpy(s.data(), mine, n);
    m->meet();
    for (int p = 0; p < size; p++) memcpy((char*)all + (size_t)p * n, m->slot[p].data(), n);
    m->meet();
  }
};

// Full mesh: rank r listens on its own port, connects to every lower rank and accepts every higher
// one. Busy-polls non-blocking sockets with TCP_NODELAY: latency over CPU time.
struct TcpTransport : Transport {
  std::vector<int> fd;

  TcpTransport(int rank_, const std::vector<std::string>& hosts) {
    rank = rank_;
    size = (int)hosts.size();
    if (rank < 0 || rank >= size) {
      fprintf(stderr, "transport: rank %d of %d\n", rank, size);
      exit(1);
    }
    fd.assign(size, -1);
    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    const int on = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_addr.s_addr = htonl(INADDR_ANY);
    me.sin_port = htons((uint16_t)atoi(port_of(hosts[rank]).c_str()));
    if (bind(ls, (sockaddr*)&me, sizeof(me)) != 0) xfer_die(("bind " + hosts[rank]).c_str());
    if (listen(ls, size) != 0) xfer_die("listen");

    // Ranks start minutes apart (each box runs its own script): keep retrying.
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    for (int p = 0; p < rank; p++) {
      addrinfo hint{}, *ai = nullptr;
      hint.ai_family = AF_INET;
      hint.ai_socktype = SOCK_STREAM;
      if (getaddrinfo(host_of(hosts[p]).c_str(), port_of(hosts[p]).c_str(), &hint, &ai) != 0) {
        fprintf(stderr, "transport: cannot resolve %s\n", hosts[p].c_str());
        exit(1);
      }
      for (;;) {
        const int s = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(s, ai->ai_addr, ai->ai_addrlen) == 0) {
          fd[p] = s;
          break;
        }
        close(s);
        if (std::chrono::steady_clock::now() > give_up) xfer_die(("connect " + hosts[p]).c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
      freeaddrinfo(ai);
      const int32_t id = rank;
      if (::send(fd[p], &id, 4, 0) != 4) xfer_die("send rank");
    }
    for (int k = rank + 1; k < size; k++) {
      const int s = accept(ls, nullptr, nullptr);
      int32_t id = -1;
      if (s < 0 || ::recv(s, &id, 4, MSG_WAITALL) != 4 || id <= rank || id >= size || fd[id] >= 0) xfer_die("accept");
      fd[id] = s;
    }
    close(ls);
    for (int p = 0; p < size; p++) {
      if (p == rank) continue;
      setsockopt(fd[p], IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
      fcntl(fd[p], F_SETFL, fcntl(fd[p], F_GETFL) | O_NONBLOCK);
    }
  }
  ~TcpTransport() override {
    for (int s : fd)
      if (s >= 0) close(s);
  }
  const char* name() const override { return "tcp"; }

  size_t try_send(int p, const void* buf, size_t n) override {
    const ssize_t k = ::send(fd[p], buf, n, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (k >= 0) return (size_t)k;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return 0;
    xfer_die("send");
  }
  size_t try_recv(int p, void* buf, size_t n) override {
    const ssize_t k = ::recv(fd[p], buf, n, MSG_DONTWAIT);
    if (k > 0) return (size_t)k;
    if (k == 0) {
      fprintf(stderr, "transport: rank %d closed the connection\n", p);
      exit(1);
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return 0;
    xfer_die("recv");
  }

  static std::string host_of(const std::string& hp) { return hp.substr(0, hp.rfind(':')); }
  static std::string port_of(const std::string& hp) {
    const size_t c = hp.rfind(':');
    if (c == std::string::npos) {
      fprintf(stderr, "transport: expected host:port, got %s\n", hp.c_str());
      exit(1);
    }
    return hp.substr(c + 1);
  }
};

inline std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> v;
  for (size_t a = 0; a <= s.size();) {
    const size_t b = std::min(s.find(',', a), s.size());
    if (b > a) v.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return v;
}

#ifdef MK_VERBS
#include "verbs.h"
#endif
#ifdef MK_ODL
#include "odl.h"
#endif

// "local" (one rank, no network), "host:port,host:port,..." (TCP, one entry per rank) or, built with -DMK_VERBS,
// "verbs:host:port,..." (RDMA), or built with -DMK_ODL, "odl:host:port,..." (OdinLink USB4); the TCP addresses
// set both up.
inline Transport* open_transport(int rank, const std::string& peers) {
  if (peers == "local") return new LocalTransport;
  if (peers.rfind("verbs:", 0) == 0) {
#ifdef MK_VERBS
    return new VerbsTransport(rank, split_list(peers.substr(6)));
#else
    fprintf(stderr, "transport: built without -DMK_VERBS\n");
    exit(1);
#endif
  }
  if (peers.rfind("odl:", 0) == 0) {
#ifdef MK_ODL
    return new OdlTransport(rank, split_list(peers.substr(4)));
#else
    fprintf(stderr, "transport: built without -DMK_ODL\n");
    exit(1);
#endif
  }
  return new TcpTransport(rank, split_list(peers));
}

// ------------------------------------------------------------------ timing
inline double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Pct {
  double p50, p90, p99, min, mean;
};
inline Pct percentiles(std::vector<double> v) {
  if (v.empty()) return {0, 0, 0, 0, 0};
  std::sort(v.begin(), v.end());
  double s = 0;
  for (double x : v) s += x;
  auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; };
  return {at(0.5), at(0.9), at(0.99), v[0], s / v.size()};
}
