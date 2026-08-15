"""Microcontroller bridge — LED matrix frames, mesh firmware, edge kernels.

Three bridges between the hoa64 lab and small hardware:

**WS2812 LED matrices.**  A W×H WS2812 ("NeoPixel") panel is one daisy-
chained shift register: each LED latches 24 bits (G8 R8 B8, MSB first,
≈0.4/0.85 µs pulses at 800 kHz) and forwards the rest downstream.  The
stream layout is therefore GRB — not RGB — and panels wired as a
zig-zag ("serpentine") raster need the odd rows reversed so logical
row-major pixels land on the right physical diode.  `pack_frame` does
both remaps and emits the raw byte stream the firmware HTTP endpoint
consumes.

**ESP-NOW RSSI tomography (ALPHA).**  ESP32 nodes broadcast ESP-NOW
beacons; every receiver logs the per-peer RSSI (via the WiFi
promiscuous rx-control block — ESP-NOW's own recv callback carries no
RSSI) and forwards its row to a gateway that serves the full n×n link
matrix over HTTP.  Moving bodies attenuate the 2.4 GHz links, so the
delta against an empty-room baseline is a coarse motion field.  RSSI is
quantized in dBm and multipath-dominated — no CSI phase, no sub-meter
resolution; treat the output as an occupancy hint, not imaging.

**Edge engine ports.**  The lab's math kernels re-templated for targets
without NumPy: `hadamard_core` (bitset Sylvester construction,
popcount-dot orthogonality verify, ±1 perturbation, one greedy
single-flip descent step on the exchange energy
E = Σ_{i≠j} G_ij², G = HHᵀ — the integer twin of the ILS max-det
descent in `hadamard.py`), `flux_map` (the domain-wall density of
`micromag.flux_map` plus the flux-tile histogram), and `terrain_fbm`
(value-noise fBm on an integer-hash lattice, smoothstep bilinear
interpolation; the hash is integer-only, the octave loop uses float or
Q16.16-friendly small multiplies, output normalized to [-1, 1]).

Every kernel is implemented twice: as a plain-Python reference
(`py_*`, cross-checked against the package in the selftest) and as
textual CircuitPython / Rust-no_std / C-baremetal templates emitted by
`export_engine`.  No NumPy anywhere on-device.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- frame packing

def _remap_index(row: int, col: int, w: int, serpentine: bool) -> int:
    """Logical (row, col) → physical chain index (odd rows reversed)."""
    if serpentine and (row & 1):
        col = w - 1 - col
    return row * w + col


def pack_frame(pixels, w: int, h: int, serpentine: bool = True) -> bytes:
    """Pack one W×H frame into the WS2812 GRB byte stream.

    ``pixels`` is a row-major list of ``(r, g, b)`` int triples 0..255.
    With ``serpentine`` the odd rows are reversed to match zig-zag panel
    wiring.  Returns exactly ``w*h*3`` bytes, G R B per LED, in chain
    order — the body contract of the firmware's ``POST /frame``.
    """
    px = list(pixels)
    if len(px) != w * h:
        raise ValueError(f"expected {w*h} pixels, got {len(px)}")
    out = bytearray(w * h * 3)
    for row in range(h):
        for col in range(w):
            r, g, b = (int(c) & 0xFF for c in px[row * w + col])
            dst = _remap_index(row, col, w, serpentine) * 3
            out[dst] = g          # WS2812 wire order is GRB
            out[dst + 1] = r
            out[dst + 2] = b
    return bytes(out)


def pack_frames(frames, w: int, h: int, serpentine: bool = True) -> bytes:
    """Concatenate `pack_frame` over a list of frames."""
    return b"".join(pack_frame(f, w, h, serpentine) for f in frames)


# ---------------------------------------------------------------- py kernels
# Plain-Python references for the edge ports.  The selftest pins these
# against hoa64.hadamard / hoa64.micromag; the target templates below
# are textual ports of THE SAME algorithms.

def py_sylvester(n: int) -> list[list[int]]:
    """Sylvester H_{2^k} as a list-of-lists ±1, bitset-free."""
    if n < 1 or (n & (n - 1)):
        raise ValueError(f"n must be a power of two, got {n}")
    H = [[1]]
    while len(H) < n:
        H = [row + row for row in H] + [row + [-v for v in row] for row in H]
    return H


def py_verify(H) -> bool:
    """Orthogonality via integer row dot products (popcount-dot twin)."""
    n = len(H)
    if n == 0 or any(len(row) != n for row in H):
        return False
    for row in H:
        if any(v not in (1, -1) for v in row):
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if sum(a * b for a, b in zip(H[i], H[j])) != 0:
                return False
    return True


def py_perturb(H, rng) -> list[list[int]]:
    """Flip one random entry (±1 sign flip), returns a new matrix."""
    n = len(H)
    i, j = rng.randrange(n), rng.randrange(n)
    out = [list(row) for row in H]
    out[i][j] = -out[i][j]
    return out


def _gram(H) -> list[list[int]]:
    n = len(H)
    return [[sum(a * b for a, b in zip(H[i], H[j])) for j in range(n)]
            for i in range(n)]


def _energy(G) -> int:
    """Exchange energy E = Σ_{i≠j} G_ij²; zero iff Hadamard."""
    n = len(G)
    return sum(G[i][j] * G[i][j] for i in range(n) for j in range(n) if i != j)


def py_ils_step(H) -> tuple[list[list[int]], bool, int, int]:
    """One greedy single-flip descent step on E = Σ_{i≠j} G_ij².

    Tries every entry flip, applies the one with the largest energy
    drop.  Returns ``(H, improved, E_before, E_after)``.  This is the
    integer-only on-device twin of the incremental Gram-matrix descent
    in `hadamard.py` (quadratic in n² flips × O(n) each — fine for the
    small orders a microcontroller can hold).
    """
    n = len(H)
    G = _gram(H)
    e0 = _energy(G)
    best = None  # (delta, i, j)
    for i in range(n):
        for j in range(n):
            # flipping H[i][j] changes row i's dots with every other row
            delta = 0
            for k in range(n):
                if k == i:
                    continue
                g = G[i][k] - 2 * H[i][j] * H[k][j]  # new dot after flip
                delta += g * g - G[i][k] * G[i][k]
            if best is None or delta < best[0]:
                best = (delta, i, j)
    if best is not None and best[0] < 0:
        _, i, j = best
        H[i][j] = -H[i][j]
        return H, True, e0, e0 + best[0]
    return H, False, e0, e0


def py_flux_map(H) -> list[list[float]]:
    """Domain-wall density W[i,j] = (2 − hb − vb)/4, toroidal bonds.

    Mirrors `micromag.flux_map` exactly: hb = H[i,j]·H[i,j+1],
    vb = H[i,j]·H[i+1,j] (cyclic boundary).  W ∈ {0, ½, 1}.
    """
    n = len(H)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            hb = H[i][j] * H[i][(j + 1) % n]
            vb = H[i][j] * H[(i + 1) % n][j]
            W[i][j] = (2.0 - hb - vb) / 4.0
    return W


def py_wall4(H) -> list[list[float]]:
    """Fraction of the 4 toroidal neighbors with opposite sign (0..1).

    The symmetric four-neighbor reading of the wall density — equals
    ½·(W_out(i,j) + W_out(i−1,j) + W_out(i,j−1) corner contributions);
    kept alongside `py_flux_map` because it is the cheaper viz on a
    small screen (one value per site, no bond bookkeeping).
    """
    n = len(H)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = H[i][j]
            opp = (s * H[i][(j + 1) % n] < 0) + (s * H[i][(j - 1) % n] < 0) \
                + (s * H[(i + 1) % n][j] < 0) + (s * H[(i - 1) % n][j] < 0)
            W[i][j] = opp / 4.0
    return W


def py_tile_hist(H, tile: int = 8) -> dict:
    """Histogram of unique t×t blocks of `py_flux_map` (H.8 test).

    On Sylvester n≥16 the flux map tessellates into exactly 4 unique
    8×8 tiles (see `micromag.flux_tiles`); returns the per-tile counts
    keyed by the flattened block as a string.
    """
    n = len(H)
    if n % tile:
        raise ValueError(f"n={n} not a multiple of tile={tile}")
    W = py_flux_map(H)
    hist: dict[str, int] = {}
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            key = ",".join(str(W[i][j]) for i in range(bi, bi + tile)
                           for j in range(bj, bj + tile))
            hist[key] = hist.get(key, 0) + 1
    return hist


# xxhash-style 32-bit lattice hash — integer only, fixed-point friendly.
_MASK32 = 0xFFFFFFFF


def _ihash(ix: int, iy: int, seed: int) -> float:
    """Hash lattice point (ix, iy) to a pseudo-random value in [-1, 1]."""
    h = (ix * 0x9E3779B1) & _MASK32
    h ^= (iy * 0x85EBCA77) & _MASK32
    h ^= (seed * 0xC2B2AE3D) & _MASK32
    h &= _MASK32
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & _MASK32
    h ^= h >> 12
    h = (h * 0x297A2D39) & _MASK32
    h ^= h >> 15
    return 2.0 * h / _MASK32 - 1.0


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def py_fbm(u: float, v: float, octaves: int = 4, seed: int = 0) -> float:
    """Value-noise fBm, output normalized to [-1, 1].

    ``u, v`` are continuous coordinates; the lattice frequency doubles
    per octave from 2 and the amplitude halves (classic 1/f fBm).  All
    lattice values come from `_ihash` (integer mix, no tables), so the
    port to fixed point only needs the smoothstep/interp chain — the
    multiply widths stay under 32 bits for lattice coords < 2¹⁵.
    """
    total = 0.0
    amp = 1.0
    norm = 0.0
    freq = 2.0
    for _ in range(octaves):
        x, y = u * freq, v * freq
        x0, y0 = math.floor(x), math.floor(y)
        tx, ty = _smooth(x - x0), _smooth(y - y0)
        n00 = _ihash(x0, y0, seed)
        n10 = _ihash(x0 + 1, y0, seed)
        n01 = _ihash(x0, y0 + 1, seed)
        n11 = _ihash(x0 + 1, y0 + 1, seed)
        n = (n00 * (1 - tx) + n10 * tx) * (1 - ty) + (n01 * (1 - tx) + n11 * tx) * ty
        total += amp * n
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


# ---------------------------------------------------------------- firmware
# C/C++/MicroPython bodies are kept as plain literals (no f-strings — the
# brace density would be unreadable); per-config values live in a small
# generated header block concatenated in front.

_LED_README = """\
# hoa64 LED matrix firmware

W×H WS2812 panel driven from the hoa64 webapp (Microcontroller tab).

## Wiring

| Panel | MCU (ESP32) | MCU (Teensy) | CircuitPython board |
|-------|-------------|--------------|---------------------|
| DIN   | GPIO {pin}  | pin {pin}    | board.D{pin}        |
| 5V    | 5V / VBUS   | 5V / VIN     | 5V                  |
| GND   | GND         | GND          | GND                 |

Add a ~470 Ω resistor in series with DIN and a 1000 µF cap across 5V/GND
for anything above a handful of LEDs.  Power injection every ~2 A of
LED current.  Panels wired as a zig-zag raster use SERPENTINE=1.

## Libraries

* ESP32 / Teensy (Arduino IDE): **FastLED** (Library Manager).
* CircuitPython: `neopixel` from the Adafruit bundle (already in most
  images); settings come from `os.getenv` (set them in `settings.toml`).

## Flash

1. Open the `.ino` (Arduino IDE, board = your ESP32/Teensy) or copy
   `code.py` to the CIRCUITPY drive.
2. Set WIFI_SSID/WIFI_PASS for station mode, or leave the SSID empty —
   the board then opens its own AP `HOA64-MATRIX` at 192.168.4.1.

## HTTP contract (port 80)

* `POST /frame` — raw body of exactly W·H·3 bytes, WS2812 GRB order,
  logical row-major (the firmware re-remaps serpentine rows).  Replies
  `200 OK`.
* `GET /state` — JSON `{{"w": W, "h": H, "frames_received": N}}`.

The hoa64 endpoints `/api/mcu/push` and `hoa64.mcu.pack_frame` speak
this contract exactly.
"""

_ESP32_LED_BODY = r"""
#include <FastLED.h>
#include <WiFi.h>

CRGB leds[NUM_LEDS];
WiFiServer server(80);
uint32_t frames_received = 0;

// logical (row, col) -> physical chain index (serpentine odd rows reversed)
static int remapIndex(int row, int col) {
  if (SERPENTINE && (row & 1)) col = WIDTH - 1 - col;
  return row * WIDTH + col;
}

static void applyFrame(const uint8_t* buf) {
  // body is GRB, logical row-major
  for (int row = 0; row < HEIGHT; row++)
    for (int col = 0; col < WIDTH; col++) {
      int src = (row * WIDTH + col) * 3;
      leds[remapIndex(row, col)] = CRGB(buf[src + 1], buf[src], buf[src + 2]);
    }
  FastLED.show();
  frames_received++;
}

static void reply(WiFiClient& client, const char* status, const char* ctype,
                  const String& body) {
  client.printf("HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %u\r\n"
                "Connection: close\r\n\r\n", status, ctype, (unsigned)body.length());
  client.print(body);
}

static void handleClient(WiFiClient& client) {
  client.setTimeout(2);
  String req = client.readStringUntil('\n');        // request line
  int contentLength = 0;
  while (true) {                                     // headers
    String line = client.readStringUntil('\n');
    if (line.length() <= 1) break;                   // blank line = end
    line.toLowerCase();
    if (line.startsWith("content-length:"))
      contentLength = line.substring(15).toInt();
  }
  if (req.startsWith("POST /frame")) {
    static uint8_t buf[NUM_LEDS * 3];
    int got = 0;
    int want = min(contentLength, (int)sizeof(buf));
    uint32_t t0 = millis();
    while (got < want && millis() - t0 < 3000) {     // blocking body read
      int r = client.read(buf + got, want - got);
      if (r > 0) got += r; else delay(1);
    }
    if (got >= NUM_LEDS * 3) {
      applyFrame(buf);
      reply(client, "200 OK", "text/plain", "ok\n");
    } else {
      reply(client, "400 Bad Request", "text/plain", "short body\n");
    }
  } else if (req.startsWith("GET /state")) {
    reply(client, "200 OK", "application/json",
          String("{\"w\":") + WIDTH + ",\"h\":" + HEIGHT +
          ",\"frames_received\":" + frames_received + "}");
  } else {
    reply(client, "404 Not Found", "text/plain", "no such path\n");
  }
  client.stop();
}

void setup() {
  FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);
  FastLED.clear(true);
  if (strlen(WIFI_SSID) > 0) {                       // station mode
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) delay(200);
  }
  if (strlen(WIFI_SSID) == 0 || WiFi.status() != WL_CONNECTED) {
    WiFi.mode(WIFI_AP);                              // fallback AP
    WiFi.softAP("HOA64-MATRIX");                     // 192.168.4.1
  }
  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  if (client) handleClient(client);
}
"""

_TEENSY_LED_BODY = r"""
#include <FastLED.h>

CRGB leds[NUM_LEDS];
uint32_t frame = 0;

// NOTE: this build is demo-only — a Teensy has no on-board WiFi.  Bolt on
// a WiFiNINA/ESP32 co-processor and mirror the ESP32 sketch's POST /frame
// handler if you want network frames; the remap contract is identical.
static int remapIndex(int row, int col) {
  if (SERPENTINE && (row & 1)) col = WIDTH - 1 - col;
  return row * WIDTH + col;
}

void setup() {
  FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);
  FastLED.clear(true);
}

void loop() {
  // built-in demo: diagonal hue wipe + sparse plasma
  for (int row = 0; row < HEIGHT; row++)
    for (int col = 0; col < WIDTH; col++) {
      uint8_t hue = (uint8_t)((row * HEIGHT + col) * 4 + frame * 3);
      if ((row + col + frame) % 8 < 3)
        leds[remapIndex(row, col)] = CHSV(hue, 255, 255);
      else
        leds[remapIndex(row, col)] = CRGB::Black;
    }
  FastLED.show();
  frame++;
  delay(60);
}
"""

_CIRCUITPY_LED_BODY = r"""
import board
import neopixel
import os
import socketpool
import wifi

W = int(os.getenv("MATRIX_W", "{w}"))
H = int(os.getenv("MATRIX_H", "{h}"))
PIN = getattr(board, "D{pin}")
BRIGHTNESS = {brightness} / 255.0
SERPENTINE = bool({serpentine})
SSID = os.getenv("WIFI_SSID", "{ssid}")
PASSWORD = os.getenv("WIFI_PASS", "{password}")

N = W * H
pixels = neopixel.NeoPixel(PIN, N, brightness=BRIGHTNESS, auto_write=False,
                           pixel_order=neopixel.GRB)


def remap(row, col):
    if SERPENTINE and (row & 1):
        col = W - 1 - col
    return row * W + col


frames_received = 0


def apply_frame(buf):
    global frames_received
    for row in range(H):
        base = row * W * 3
        for col in range(W):
            src = base + col * 3
            # body is GRB logical row-major; neopixel GRB takes (r, g, b)
            pixels[remap(row, col)] = (buf[src + 1], buf[src], buf[src + 2])
    pixels.show()
    frames_received += 1


if SSID:
    wifi.radio.connect(SSID, PASSWORD)
else:
    wifi.radio.start_ap("HOA64-MATRIX")
print("listening on", wifi.radio.ipv4_address)

pool = socketpool.SocketPool(wifi.radio)
srv = pool.socket(pool.AF_INET, pool.SOCK_STREAM)
srv.setblocking(False)
srv.bind(("0.0.0.0", 80))
srv.listen(1)

while True:
    conn = None
    try:
        conn, _addr = srv.accept()
    except OSError:
        continue
    conn.settimeout(3.0)
    try:
        req = b""
        while b"\r\n\r\n" not in req:              # request line + headers
            chunk = conn.recv(256)
            if not chunk:
                break
            req += chunk
        line, _, rest = req.partition(b"\r\n")
        headers, _, body = rest.partition(b"\r\n\r\n")
        clen = 0
        for h in headers.split(b"\r\n"):
            if h.lower().startswith(b"content-length:"):
                clen = int(h.split(b":")[1])
        while len(body) < clen:                    # blocking body read
            body += conn.recv(min(1024, clen - len(body)))
        if line.startswith(b"POST /frame"):
            if len(body) >= N * 3:
                apply_frame(body[: N * 3])
                conn.send(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n"
                          b"Connection: close\r\n\r\nok\n")
            else:
                conn.send(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n"
                          b"Connection: close\r\n\r\n")
        elif line.startswith(b"GET /state"):
            payload = ('{{"w": %d, "h": %d, "frames_received": %d}}'
                       % (W, H, frames_received)).encode()
            conn.send(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                      b"Content-Length: " + str(len(payload)).encode() +
                      b"\r\nConnection: close\r\n\r\n" + payload)
        else:
            conn.send(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                      b"Connection: close\r\n\r\n")
    except (OSError, ValueError) as e:
        print("request error:", e)
    finally:
        conn.close()
"""


def _arduino_header(cfg: dict) -> str:
    ssid = cfg.get("ssid") or ""
    password = cfg.get("password") or ""
    lines = [
        "// hoa64 LED matrix firmware — generated by hoa64.mcu.led_firmware",
        f"#define WIDTH {cfg['w']}",
        f"#define HEIGHT {cfg['h']}",
        f"#define DATA_PIN {cfg['pin']}",
        f"#define NUM_LEDS (WIDTH * HEIGHT)",
        f"#define BRIGHTNESS {cfg['brightness']}",
        f"#define SERPENTINE {1 if cfg['serpentine'] else 0}",
        f'#define WIFI_SSID "{ssid}"',
        f'#define WIFI_PASS "{password}"',
        "",
    ]
    return "\n".join(lines)


def led_firmware(cfg: dict) -> dict[str, str]:
    """Generate W×H WS2812 matrix firmware as filename → content.

    cfg keys: ``board`` ∈ {esp32, teensy, circuitpython}, ``w``/``h``
    (1..64), ``pin`` (default GPIO 13 on ESP32, 2 on Teensy/CircuitPython),
    ``serpentine``, ``ssid``/``password`` (optional — empty ssid means the
    board opens its own AP ``HOA64-MATRIX``), ``brightness`` (0..255,
    default 64).  Always includes a README.md with wiring and the HTTP
    frame contract.
    """
    board = cfg.get("board", "esp32")
    if board not in ("esp32", "teensy", "circuitpython"):
        raise ValueError(f"unknown board {board!r}")
    w, h = int(cfg.get("w", 16)), int(cfg.get("h", 16))
    if not (1 <= w <= 64 and 1 <= h <= 64):
        raise ValueError("w/h must be 1..64")
    pin = int(cfg.get("pin", 13 if board == "esp32" else 2))
    brightness = int(cfg.get("brightness", 64))
    if not (0 <= brightness <= 255):
        raise ValueError("brightness must be 0..255")
    full = {
        "w": w, "h": h, "pin": pin, "brightness": brightness,
        "serpentine": bool(cfg.get("serpentine", True)),
        "ssid": cfg.get("ssid") or "", "password": cfg.get("password") or "",
    }
    if board == "esp32":
        files = {"hoa64_matrix_esp32.ino": _arduino_header(full) + _ESP32_LED_BODY}
    elif board == "teensy":
        files = {"hoa64_matrix_teensy.ino": _arduino_header(full) + _TEENSY_LED_BODY}
    else:
        files = {"code.py": _CIRCUITPY_LED_BODY.format(
            w=w, h=h, pin=pin, brightness=brightness,
            serpentine=1 if full["serpentine"] else 0,
            ssid=full["ssid"], password=full["password"])}
    files["README.md"] = _LED_README.format(pin=pin)
    return files


# ---------------------------------------------------------------- mesh firmware

_MESH_README = """\
# hoa64 WiFi mesh field firmware (ALPHA)

ESP-NOW RSSI tomography: N nodes broadcast beacons every 500 ms, each
node logs the per-peer RSSI and forwards its row to the gateway, which
serves the n×n link matrix over HTTP.  Coarse occupancy sensing only —
RSSI is dBm-quantized and multipath-dominated; no CSI, no phase.

## Flash

1. Open `hoa64_mesh.ino` (Arduino IDE, board = ESP32).
2. Set `#define NODE_ID` uniquely per node: 0..{n_max} ({n_nodes} nodes).
3. Node {gateway_id} is the gateway: it opens AP `HOA64-MESH` at
   192.168.4.1 and serves `GET /mesh`.

## HTTP contract (gateway, port 80)

`GET /mesh` →
`{{"t": <ms>, "n": {n_nodes}, "rssi_dbm": [[.. n×n ..]]}}` —
diagonal is 0, `-128` means no link heard in the last epoch.

Collect with `POST /api/mcu/mesh/collect {{"host": "192.168.4.1"}}` from
the hoa64 webapp.
"""

_MESH_INO_BODY = r"""
#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

// ---- hoa64 mesh field (ALPHA) -------------------------------------------
// Every node broadcasts a beacon {magic, node_id, seq} via ESP-NOW every
// 500 ms.  RSSI is NOT available in the ESP-NOW recv callback, so a WiFi
// promiscuous callback sniffs the same beacon frames and records
// wifi_pkt_rx_ctrl_t.rssi per peer — that is the standard workaround, and
// it costs us the ESP-NOW CRC/magic filtering, so we re-check the magic
// bytes in the promiscuous payload manually.  Limitation: the promiscuous
// payload body layout depends on the IDF version; we scan the first 64
// bytes for the magic.  RSSI is per-beacon, ~1 Hz update, dBm-quantized.

#define MAGIC0 0x68
#define MAGIC1 0x6f  // "ho"
#define MAX_NODES 12
#define NO_LINK (-128)

static volatile int8_t rssi_row[MAX_NODES];
static uint8_t seq = 0;
static uint32_t last_tx = 0;

typedef struct __attribute__((packed)) {
  uint8_t magic0, magic1, node_id, seq;
} beacon_t;

typedef struct __attribute__((packed)) {
  uint8_t magic0, magic1, node_id;
  int8_t row[MAX_NODES];
} row_report_t;

static uint8_t bcast[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// promiscuous sniff: find the beacon magic, take the frame RSSI
static void sniffer(void* buf, wifi_promiscuous_pkt_type_t type) {
  if (type != WIFI_PKT_DATA) return;
  const wifi_promiscuous_pkt_t* pkt = (const wifi_promiscuous_pkt_t*)buf;
  const uint8_t* p = pkt->payload;
  int len = pkt->rx_ctrl.sig_len;
  for (int i = 0; i + 4 <= len && i < 64; i++) {
    if (p[i] == MAGIC0 && p[i + 1] == MAGIC1 && p[i + 2] < MAX_NODES) {
      rssi_row[p[i + 2]] = (int8_t)pkt->rx_ctrl.rssi;
      return;
    }
  }
}

// ESP-NOW recv: row reports for the gateway
static void onRecv(const uint8_t* mac, const uint8_t* data, int len) {
  if (NODE_ID != GATEWAY_ID) return;
  if (len == sizeof(row_report_t) && data[0] == MAGIC0 && data[1] == MAGIC1
      && data[2] < MAX_NODES) {
    const row_report_t* rep = (const row_report_t*)data;
    for (int j = 0; j < MAX_NODES; j++)
      if (rep->row[j] != NO_LINK) rssi_matrix[rep->node_id][j] = rep->row[j];
  }
}

void setup() {
  for (int i = 0; i < MAX_NODES; i++) rssi_row[i] = NO_LINK;
  for (int i = 0; i < MAX_NODES; i++)
    for (int j = 0; j < MAX_NODES; j++) rssi_matrix[i][j] = NO_LINK;
  WiFi.mode(WIFI_AP_STA);                    // AP for the gateway HTTP
  if (NODE_ID == GATEWAY_ID) WiFi.softAP("HOA64-MESH");   // 192.168.4.1
  esp_now_init();
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, bcast, 6);
  peer.channel = 0;
  esp_now_add_peer(&peer);
  esp_now_register_recv_cb(onRecv);
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_promiscuous_rx_cb(sniffer);
  if (NODE_ID == GATEWAY_ID) server.begin();
}

static void serveMesh(WiFiClient& client) {
  String body = "{\"t\":" + String(millis()) + ",\"n\":" + String(N_NODES) +
                ",\"rssi_dbm\":[";
  for (int i = 0; i < N_NODES; i++) {
    body += "[";
    for (int j = 0; j < N_NODES; j++) {
      body += String(i == j ? 0 : rssi_matrix[i][j]);
      if (j + 1 < N_NODES) body += ",";
    }
    body += "]";
    if (i + 1 < N_NODES) body += ",";
  }
  body += "]}";
  client.printf("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Content-Length: %u\r\nConnection: close\r\n\r\n",
                (unsigned)body.length());
  client.print(body);
}

void loop() {
  uint32_t now = millis();
  if (now - last_tx >= 500) {                // 2 Hz beacon + report
    last_tx = now;
    beacon_t b = {MAGIC0, MAGIC1, (uint8_t)NODE_ID, seq++};
    esp_now_send(bcast, (const uint8_t*)&b, sizeof(b));
    if (NODE_ID == GATEWAY_ID) {
      for (int j = 0; j < MAX_NODES; j++)
        rssi_matrix[NODE_ID][j] = rssi_row[j];
    } else {
      row_report_t rep = {MAGIC0, MAGIC1, (uint8_t)NODE_ID, {}};
      for (int j = 0; j < MAX_NODES; j++) rep.row[j] = rssi_row[j];
      esp_now_send(bcast, (const uint8_t*)&rep, sizeof(rep));
    }
  }
  if (NODE_ID == GATEWAY_ID) {
    WiFiClient client = server.available();
    if (client) {
      client.setTimeout(2);
      String req = client.readStringUntil('\n');
      while (client.readStringUntil('\n').length() > 1) {}  // skip headers
      if (req.startsWith("GET /mesh")) serveMesh(client);
      client.stop();
    }
  }
}
"""


def mesh_firmware(cfg: dict) -> dict[str, str]:
    """Generate the ESP-NOW mesh-field firmware (one parameterized sketch).

    cfg keys: ``n_nodes`` (2..12), ``gateway_id`` (0..n_nodes−1).  The
    same ``.ino`` is flashed to every node — only ``#define NODE_ID`` at
    the top changes per device.  The gateway node opens AP
    ``HOA64-MESH`` (192.168.4.1) and serves ``GET /mesh``.
    """
    n = int(cfg.get("n_nodes", 4))
    if not (2 <= n <= 12):
        raise ValueError("n_nodes must be 2..12")
    gw = int(cfg.get("gateway_id", 0))
    if not (0 <= gw < n):
        raise ValueError("gateway_id must be 0..n_nodes-1")
    header = "\n".join([
        "// hoa64 mesh field firmware — generated by hoa64.mcu.mesh_firmware",
        "// FLASH ONE COPY PER NODE — set NODE_ID uniquely: 0..%d" % (n - 1),
        "#define NODE_ID 0  // <— set per node before flashing",
        f"#define GATEWAY_ID {gw}",
        f"#define N_NODES {n}",
        "#include <WiFi.h>",
        "static WiFiServer server(80);",
        "static volatile int8_t rssi_matrix[12][12];",
        "",
    ])
    return {
        "hoa64_mesh.ino": header + _MESH_INO_BODY,
        "README.md": _MESH_README.format(n_nodes=n, n_max=n - 1, gateway_id=gw),
    }


# ---------------------------------------------------------------- edge export

ENGINES = ("hadamard_core", "flux_map", "terrain_fbm")
TARGETS = ("circuitpython", "rust_no_std", "c_baremetal")

# --- CircuitPython templates (pure Python + math; ulab optional) -----------

_CP_HADAMARD = '''\
"""hadamard_core — bitset-free Sylvester + integer descent (hoa64 port).

Pure Python + `math` only — no NumPy.  Drop on any CircuitPython board;
`ulab` can accelerate `verify` but is not required.  Same algorithms as
hoa64.mcu.py_sylvester / py_verify / py_perturb / py_ils_step.
"""

import random


def sylvester(n):
    """Sylvester H_(2^k) as a list-of-lists ±1."""
    if n < 1 or (n & (n - 1)):
        raise ValueError("n must be a power of two")
    H = [[1]]
    while len(H) < n:
        H = [row + row for row in H] + [row + [-v for v in row] for row in H]
    return H


def verify(H):
    """True iff H is ±1 with pairwise-orthogonal rows (integer dots)."""
    n = len(H)
    if n == 0 or any(len(row) != n for row in H):
        return False
    for row in H:
        if any(v not in (1, -1) for v in row):
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if sum(a * b for a, b in zip(H[i], H[j])) != 0:
                return False
    return True


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def energy(H):
    """E = sum of squared off-diagonal Gram entries; 0 iff Hadamard."""
    n = len(H)
    e = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                g = _dot(H[i], H[j])
                e += g * g
    return e


def perturb(H, rng=None):
    """Flip one random entry; returns a new matrix."""
    rng = rng or random
    n = len(H)
    i, j = rng.randrange(n), rng.randrange(n)
    out = [list(row) for row in H]
    out[i][j] = -out[i][j]
    return out


def ils_step(H):
    """One greedy single-flip descent on E; returns (H, improved, e0, e1)."""
    n = len(H)
    G = [[_dot(H[i], H[j]) for j in range(n)] for i in range(n)]
    e0 = sum(G[i][j] * G[i][j] for i in range(n) for j in range(n) if i != j)
    best = None
    for i in range(n):
        for j in range(n):
            delta = 0
            for k in range(n):
                if k == i:
                    continue
                g = G[i][k] - 2 * H[i][j] * H[k][j]
                delta += g * g - G[i][k] * G[i][k]
            if best is None or delta < best[0]:
                best = (delta, i, j)
    if best is not None and best[0] < 0:
        _, i, j = best
        H[i][j] = -H[i][j]
        return H, True, e0, e0 + best[0]
    return H, False, e0, e0


if __name__ == "__main__":
    H = sylvester(16)
    print("sylvester(16) hadamard:", verify(H))
    H2 = perturb(perturb(H))
    print("after 2 flips E =", energy(H2))
    H2, ok, e0, e1 = ils_step(H2)
    print("ils step:", ok, e0, "->", e1)
'''

_CP_FLUX = '''\
"""flux_map — domain-wall density + tile histogram (hoa64 port).

Mirrors hoa64.micromag.flux_map: W[i,j] = (2 - hb - vb)/4 on toroidal
bonds, W in {0, 0.5, 1}.  wall4 is the symmetric 4-neighbor fraction.
tile_hist counts unique t×t flux blocks (H.8 tessellation test).
Pure Python only.
"""


def flux_map(H):
    """W[i,j] = (2 - H[i,j]*H[i,j+1] - H[i,j]*H[i+1,j])/4, toroidal."""
    n = len(H)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            hb = H[i][j] * H[i][(j + 1) % n]
            vb = H[i][j] * H[(i + 1) % n][j]
            W[i][j] = (2.0 - hb - vb) / 4.0
    return W


def wall4(H):
    """Fraction of the 4 toroidal neighbors with opposite sign."""
    n = len(H)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = H[i][j]
            opp = (s * H[i][(j + 1) % n] < 0) + (s * H[i][(j - 1) % n] < 0)
            opp += (s * H[(i + 1) % n][j] < 0) + (s * H[(i - 1) % n][j] < 0)
            W[i][j] = opp / 4.0
    return W


def tile_hist(H, tile=8):
    """Histogram {flattened t×t flux block: count} — 4 entries on H.16."""
    n = len(H)
    if n % tile:
        raise ValueError("n must be a multiple of tile")
    W = flux_map(H)
    hist = {}
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            key = ",".join(str(W[i][j]) for i in range(bi, bi + tile)
                           for j in range(bj, bj + tile))
            hist[key] = hist.get(key, 0) + 1
    return hist
'''

_CP_FBM = '''\
"""terrain_fbm — value-noise fBm on an integer-hash lattice (hoa64 port).

The lattice hash is integer-only (xxhash-style 32-bit mix); the octave
loop needs only small multiplies — friendly to Q16.16 if floats are
scarce.  Output is normalized to [-1, 1].  Same algorithm as
hoa64.mcu.py_fbm.
"""

import math

_MASK32 = 0xFFFFFFFF


def ihash(ix, iy, seed):
    """Hash lattice point (ix, iy) to a value in [-1, 1]."""
    h = (ix * 0x9E3779B1) & _MASK32
    h ^= (iy * 0x85EBCA77) & _MASK32
    h ^= (seed * 0xC2B2AE3D) & _MASK32
    h &= _MASK32
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & _MASK32
    h ^= h >> 12
    h = (h * 0x297A2D39) & _MASK32
    h ^= h >> 15
    return 2.0 * h / _MASK32 - 1.0


def fbm(u, v, octaves=4, seed=0):
    """Value-noise fBm, output in [-1, 1]."""
    total = 0.0
    amp = 1.0
    norm = 0.0
    freq = 2.0
    for _ in range(octaves):
        x, y = u * freq, v * freq
        x0, y0 = math.floor(x), math.floor(y)
        tx = (x - x0) ** 2 * (3.0 - 2.0 * (x - x0))
        ty = (y - y0) ** 2 * (3.0 - 2.0 * (y - y0))
        n00 = ihash(x0, y0, seed)
        n10 = ihash(x0 + 1, y0, seed)
        n01 = ihash(x0, y0 + 1, seed)
        n11 = ihash(x0 + 1, y0 + 1, seed)
        n = (n00 * (1 - tx) + n10 * tx) * (1 - ty) + (n01 * (1 - tx) + n11 * tx) * ty
        total += amp * n
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm
'''


# --- Rust no_std templates --------------------------------------------------

_RUST_HADAMARD = '''\
#![no_std]
//! hadamard_core — integer-only Hadamard kernels (hoa64 edge port).
//!
//! Same algorithms as hoa64.mcu.py_sylvester / py_verify / py_perturb /
//! py_ils_step.  All buffers are caller-provided row-major i8 slices.

/// xorshift32 PRNG — no_std friendly replacement for rand.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(if seed == 0 { 0x9E3779B9 } else { seed })
    }
    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// Sylvester H_(2^k) into `out` (n*n row-major).  false if n is not a
/// power of two or `out` is too small.
pub fn sylvester(n: usize, out: &mut [i8]) -> bool {
    if n < 1 || (n & (n - 1)) != 0 || out.len() < n * n {
        return false;
    }
    out[0] = 1;
    let mut m = 1; // current matrix size, rows stored contiguously
    while m < n {
        // new row i+m = old row i followed by its negation; new row i =
        // old row i doubled.  Rebuild into the tail then copy back.
        for i in (0..m).rev() {
            for j in (0..m).rev() {
                let v = out[i * m + j];
                out[i * 2 * m + j] = v;
                out[i * 2 * m + j + m] = v;
                out[(i + m) * 2 * m + j] = v;
                out[(i + m) * 2 * m + j + m] = -v;
            }
        }
        m *= 2;
    }
    true
}

fn dot(h: &[i8], n: usize, i: usize, j: usize) -> i32 {
    let mut s = 0i32;
    for k in 0..n {
        s += h[i * n + k] as i32 * h[j * n + k] as i32;
    }
    s
}

/// True iff ±1 with pairwise-orthogonal rows (integer dot products).
pub fn verify(h: &[i8], n: usize) -> bool {
    if n == 0 || h.len() < n * n {
        return false;
    }
    for v in h[..n * n].iter() {
        if *v != 1 && *v != -1 {
            return false;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if dot(h, n, i, j) != 0 {
                return false;
            }
        }
    }
    true
}

/// Flip one pseudo-random entry in place.
pub fn perturb(h: &mut [i8], n: usize, rng: &mut Rng) {
    let i = (rng.next() as usize) % n;
    let j = (rng.next() as usize) % n;
    h[i * n + j] = -h[i * n + j];
}

/// E = sum of squared off-diagonal Gram entries; 0 iff Hadamard.
pub fn energy(h: &[i8], n: usize) -> i64 {
    let mut e = 0i64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let g = dot(h, n, i, j) as i64;
                e += g * g;
            }
        }
    }
    e
}

/// One greedy single-flip descent step on E.
/// Returns (improved, e_before, e_after).
pub fn ils_step(h: &mut [i8], n: usize) -> (bool, i64, i64) {
    let e0 = energy(h, n);
    let mut best: Option<(i64, usize, usize)> = None;
    for i in 0..n {
        for j in 0..n {
            let mut delta = 0i64;
            for k in 0..n {
                if k == i {
                    continue;
                }
                let g0 = dot(h, n, i, k) as i64;
                let g1 = g0 - 2 * h[i * n + j] as i64 * h[k * n + j] as i64;
                delta += g1 * g1 - g0 * g0;
            }
            if best.map_or(true, |(b, _, _)| delta < b) {
                best = Some((delta, i, j));
            }
        }
    }
    if let Some((d, i, j)) = best {
        if d < 0 {
            h[i * n + j] = -h[i * n + j];
            return (true, e0, e0 + d);
        }
    }
    (false, e0, e0)
}
'''

_RUST_FLUX = '''\
#![no_std]
//! flux_map — domain-wall density + tile census (hoa64 edge port).
//!
//! Mirrors hoa64.micromag.flux_map: W = (2 - hb - vb)/4 on toroidal
//! bonds.  Wall densities are stored as u8 quarters (4*W in 0..=4) so
//! no floats are needed.  `tile_count` is the allocation-free twin of
//! len(hoa64.mcu.py_tile_hist(...)) — unique t×t blocks by pairwise
//! comparison.

/// 4*W[i,j] = 2 - hb - vb (hb, vb the toroidal outgoing bonds).
pub fn flux_map4(h: &[i8], n: usize, w4: &mut [u8]) {
    for i in 0..n {
        for j in 0..n {
            let s = h[i * n + j];
            let hb = s * h[i * n + (j + 1) % n];
            let vb = s * h[((i + 1) % n) * n + j];
            w4[i * n + j] = (2 - hb - vb) as u8;
        }
    }
}

/// Fraction of the 4 toroidal neighbors with opposite sign, ×4.
pub fn wall4x4(h: &[i8], n: usize, w4: &mut [u8]) {
    for i in 0..n {
        for j in 0..n {
            let s = h[i * n + j];
            let mut opp = 0u8;
            if s * h[i * n + (j + 1) % n] < 0 { opp += 1; }
            if s * h[i * n + (j + n - 1) % n] < 0 { opp += 1; }
            if s * h[((i + 1) % n) * n + j] < 0 { opp += 1; }
            if s * h[((i + n - 1) % n) * n + j] < 0 { opp += 1; }
            w4[i * n + j] = opp;
        }
    }
}

/// Number of unique tile×tile flux blocks (4 on Sylvester H.16 with
/// tile=8 — the H.8 tessellation).
pub fn tile_count(h: &[i8], n: usize, tile: usize, w4: &mut [u8]) -> usize {
    flux_map4(h, n, w4);
    let nb = n / tile;
    let mut unique = 0usize;
    for b in 0..nb * nb {
        let mut seen = false;
        for a in 0..b {
            let mut same = true;
            for di in 0..tile {
                for dj in 0..tile {
                    let pa = ((a / nb) * tile + di) * n + (a % nb) * tile + dj;
                    let pb = ((b / nb) * tile + di) * n + (b % nb) * tile + dj;
                    if w4[pa] != w4[pb] {
                        same = false;
                        break;
                    }
                }
                if !same { break; }
            }
            if same {
                seen = true;
                break;
            }
        }
        if !seen {
            unique += 1;
        }
    }
    unique
}
'''

_RUST_FBM = '''\
#![no_std]
//! terrain_fbm — value-noise fBm on an integer-hash lattice (hoa64 port).
//!
//! The lattice hash is integer-only (xxhash-style 32-bit mix); f32 is
//! used for the smoothstep/interp chain, with libm::floorf for no_std.
//! Output is normalized to [-1, 1].  Same algorithm as hoa64.mcu.py_fbm.

use libm::floorf;

/// Hash lattice point (ix, iy, seed) to a value in [-1, 1].
pub fn ihash(ix: i32, iy: i32, seed: i32) -> f32 {
    let mut h = (ix as u32).wrapping_mul(0x9E3779B1);
    h ^= (iy as u32).wrapping_mul(0x85EBCA77);
    h ^= (seed as u32).wrapping_mul(0xC2B2AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A2D39);
    h ^= h >> 15;
    2.0 * (h as f32) / (0xFFFFFFFFu32 as f32) - 1.0
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Value-noise fBm, output in [-1, 1].
pub fn fbm(u: f32, v: f32, octaves: u32, seed: i32) -> f32 {
    let mut total = 0.0f32;
    let mut amp = 1.0f32;
    let mut norm = 0.0f32;
    let mut freq = 2.0f32;
    for _ in 0..octaves {
        let x = u * freq;
        let y = v * freq;
        let x0 = floorf(x) as i32;
        let y0 = floorf(y) as i32;
        let tx = smooth(x - x0 as f32);
        let ty = smooth(y - y0 as f32);
        let n00 = ihash(x0, y0, seed);
        let n10 = ihash(x0 + 1, y0, seed);
        let n01 = ihash(x0, y0 + 1, seed);
        let n11 = ihash(x0 + 1, y0 + 1, seed);
        let n = (n00 * (1.0 - tx) + n10 * tx) * (1.0 - ty)
            + (n01 * (1.0 - tx) + n11 * tx) * ty;
        total += amp * n;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    total / norm
}
'''

_RUST_CARGO_PLAIN = '''\
[package]
name = "{engine}"
version = "0.1.0"
edition = "2021"

[dependencies]
'''

_RUST_CARGO_LIBM = '''\
[package]
name = "{engine}"
version = "0.1.0"
edition = "2021"

[dependencies]
libm = "*"
'''


# --- C bare-metal templates -------------------------------------------------

_C_HADAMARD_H = '''\
#ifndef HADAMARD_CORE_H
#define HADAMARD_CORE_H

/* hadamard_core — integer-only Hadamard kernels (hoa64 edge port).
 * Buffers are caller-provided row-major int8 n*n slices; no malloc. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { uint32_t state; } h64_rng; /* xorshift32 */

void h64_rng_seed(h64_rng* rng, uint32_t seed);
int  h64_sylvester(int n, int8_t* out);           /* 0 = ok, -1 = bad n  */
int  h64_verify(const int8_t* h, int n);          /* 1 = hadamard        */
void h64_perturb(int8_t* h, int n, h64_rng* rng);
int64_t h64_energy(const int8_t* h, int n);       /* 0 iff hadamard      */
int  h64_ils_step(int8_t* h, int n, int64_t* e0, int64_t* e1);

#ifdef __cplusplus
}
#endif
#endif /* HADAMARD_CORE_H */
'''

_C_HADAMARD_C = '''\
#include "hadamard_core.h"

/* Same algorithms as hoa64.mcu.py_sylvester / py_verify / py_perturb /
 * py_ils_step — integer-only, no heap. */

void h64_rng_seed(h64_rng* rng, uint32_t seed) {
    rng->state = seed ? seed : 0x9E3779B9u;
}

static uint32_t rng_next(h64_rng* rng) {
    uint32_t x = rng->state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng->state = x;
    return x;
}

int h64_sylvester(int n, int8_t* out) {
    if (n < 1 || (n & (n - 1)) != 0) return -1;
    out[0] = 1;
    for (int m = 1; m < n; m *= 2)
        for (int i = m - 1; i >= 0; i--)
            for (int j = m - 1; j >= 0; j--) {
                int8_t v = out[i * m + j];
                out[i * 2 * m + j] = v;
                out[i * 2 * m + j + m] = v;
                out[(i + m) * 2 * m + j] = v;
                out[(i + m) * 2 * m + j + m] = -v;
            }
    return 0;
}

static int32_t dot(const int8_t* h, int n, int i, int j) {
    int32_t s = 0;
    for (int k = 0; k < n; k++) s += h[i * n + k] * h[j * n + k];
    return s;
}

int h64_verify(const int8_t* h, int n) {
    if (n < 1) return 0;
    for (int k = 0; k < n * n; k++)
        if (h[k] != 1 && h[k] != -1) return 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (dot(h, n, i, j) != 0) return 0;
    return 1;
}

void h64_perturb(int8_t* h, int n, h64_rng* rng) {
    int i = (int)(rng_next(rng) % (uint32_t)n);
    int j = (int)(rng_next(rng) % (uint32_t)n);
    h[i * n + j] = -h[i * n + j];
}

int64_t h64_energy(const int8_t* h, int n) {
    int64_t e = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) {
                int64_t g = dot(h, n, i, j);
                e += g * g;
            }
    return e;
}

int h64_ils_step(int8_t* h, int n, int64_t* e0, int64_t* e1) {
    *e0 = h64_energy(h, n);
    int64_t best = 0; int bi = -1, bj = -1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int64_t delta = 0;
            for (int k = 0; k < n; k++) {
                if (k == i) continue;
                int64_t g0 = dot(h, n, i, k);
                int64_t g1 = g0 - 2 * (int64_t)h[i * n + j] * h[k * n + j];
                delta += g1 * g1 - g0 * g0;
            }
            if (bi < 0 || delta < best) { best = delta; bi = i; bj = j; }
        }
    if (bi >= 0 && best < 0) {
        h[bi * n + bj] = -h[bi * n + bj];
        *e1 = *e0 + best;
        return 1;
    }
    *e1 = *e0;
    return 0;
}
'''

_C_FLUX_H = '''\
#ifndef FLUX_MAP_H
#define FLUX_MAP_H

/* flux_map — domain-wall density + tile census (hoa64 edge port).
 * Wall densities are u8 quarters (4*W in 0..=4); no floats, no malloc. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void h64_flux_map4(const int8_t* h, int n, uint8_t* w4); /* 4*W, 2-bond  */
void h64_wall4x4(const int8_t* h, int n, uint8_t* w4);   /* 4-neighbor   */
int  h64_tile_count(const int8_t* h, int n, int tile, uint8_t* w4);

#ifdef __cplusplus
}
#endif
#endif /* FLUX_MAP_H */
'''

_C_FLUX_C = '''\
#include "flux_map.h"

/* Mirrors hoa64.micromag.flux_map: W = (2 - hb - vb)/4 on toroidal
 * bonds, stored as u8 quarters. */

void h64_flux_map4(const int8_t* h, int n, uint8_t* w4) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int8_t s = h[i * n + j];
            int hb = s * h[i * n + (j + 1) % n];
            int vb = s * h[((i + 1) % n) * n + j];
            w4[i * n + j] = (uint8_t)(2 - hb - vb);
        }
}

void h64_wall4x4(const int8_t* h, int n, uint8_t* w4) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int8_t s = h[i * n + j];
            int opp = 0;
            opp += s * h[i * n + (j + 1) % n] < 0;
            opp += s * h[i * n + (j + n - 1) % n] < 0;
            opp += s * h[((i + 1) % n) * n + j] < 0;
            opp += s * h[((i + n - 1) % n) * n + j] < 0;
            w4[i * n + j] = (uint8_t)opp;
        }
}

int h64_tile_count(const int8_t* h, int n, int tile, uint8_t* w4) {
    /* unique tile×tile flux blocks by pairwise compare — 4 on H.16/8 */
    h64_flux_map4(h, n, w4);
    int nb = n / tile, unique = 0;
    for (int b = 0; b < nb * nb; b++) {
        int seen = 0;
        for (int a = 0; a < b && !seen; a++) {
            int same = 1;
            for (int di = 0; di < tile && same; di++)
                for (int dj = 0; dj < tile; dj++) {
                    int pa = ((a / nb) * tile + di) * n + (a % nb) * tile + dj;
                    int pb = ((b / nb) * tile + di) * n + (b % nb) * tile + dj;
                    if (w4[pa] != w4[pb]) { same = 0; break; }
                }
            if (same) seen = 1;
        }
        if (!seen) unique++;
    }
    return unique;
}
'''

_C_FBM_H = '''\
#ifndef TERRAIN_FBM_H
#define TERRAIN_FBM_H

/* terrain_fbm — value-noise fBm on an integer-hash lattice (hoa64 port).
 * The lattice hash is integer-only; float is used for smoothstep/interp
 * (Q16.16-friendly).  Output is normalized to [-1, 1]. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float h64_ihash(int32_t ix, int32_t iy, int32_t seed); /* in [-1, 1]    */
float h64_fbm(float u, float v, int octaves, int32_t seed);

#ifdef __cplusplus
}
#endif
#endif /* TERRAIN_FBM_H */
'''

_C_FBM_C = '''\
#include "terrain_fbm.h"
#include <math.h>

/* Same algorithm as hoa64.mcu.py_fbm — xxhash-style 32-bit lattice mix,
 * smoothstep bilinear value noise, 1/f octave stack, [-1, 1] output. */

float h64_ihash(int32_t ix, int32_t iy, int32_t seed) {
    uint32_t h = (uint32_t)ix * 0x9E3779B1u;
    h ^= (uint32_t)iy * 0x85EBCA77u;
    h ^= (uint32_t)seed * 0xC2B2AE3Du;
    h ^= h >> 15; h *= 0x2C1B3C6Du;
    h ^= h >> 12; h *= 0x297A2D39u;
    h ^= h >> 15;
    return 2.0f * (float)h / (float)0xFFFFFFFFu - 1.0f;
}

static float smooth(float t) { return t * t * (3.0f - 2.0f * t); }

float h64_fbm(float u, float v, int octaves, int32_t seed) {
    float total = 0.0f, amp = 1.0f, norm = 0.0f, freq = 2.0f;
    for (int k = 0; k < octaves; k++) {
        float x = u * freq, y = v * freq;
        int32_t x0 = (int32_t)floorf(x), y0 = (int32_t)floorf(y);
        float tx = smooth(x - (float)x0), ty = smooth(y - (float)y0);
        float n00 = h64_ihash(x0, y0, seed);
        float n10 = h64_ihash(x0 + 1, y0, seed);
        float n01 = h64_ihash(x0, y0 + 1, seed);
        float n11 = h64_ihash(x0 + 1, y0 + 1, seed);
        float n = (n00 * (1.0f - tx) + n10 * tx) * (1.0f - ty)
                + (n01 * (1.0f - tx) + n11 * tx) * ty;
        total += amp * n;
        norm += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return total / norm;
}
'''


def export_engine(engine: str, target: str) -> dict[str, str]:
    """Template port of one math kernel to one edge target.

    The generated files are textual ports of the ``py_*`` references in
    this module — no NumPy on-device.  ``circuitpython`` ships a single
    ``.py``; ``rust_no_std`` a Cargo project (``libm`` only for fBm);
    ``c_baremetal`` a ``.c``/``.h`` pair with caller-provided buffers.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected {ENGINES}")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected {TARGETS}")
    if target == "circuitpython":
        body = {"hadamard_core": _CP_HADAMARD, "flux_map": _CP_FLUX,
                "terrain_fbm": _CP_FBM}[engine]
        return {f"{engine}.py": body}
    if target == "rust_no_std":
        body = {"hadamard_core": _RUST_HADAMARD, "flux_map": _RUST_FLUX,
                "terrain_fbm": _RUST_FBM}[engine]
        cargo = (_RUST_CARGO_LIBM if engine == "terrain_fbm"
                 else _RUST_CARGO_PLAIN).format(engine=engine)
        return {"Cargo.toml": cargo, "lib.rs": body}
    name = {"hadamard_core": "hadamard_core", "flux_map": "flux_map",
            "terrain_fbm": "terrain_fbm"}[engine]
    c_body = {"hadamard_core": _C_HADAMARD_C, "flux_map": _C_FLUX_C,
              "terrain_fbm": _C_FBM_C}[engine]
    h_body = {"hadamard_core": _C_HADAMARD_H, "flux_map": _C_FLUX_H,
              "terrain_fbm": _C_FBM_H}[engine]
    return {f"{name}.c": c_body, f"{name}.h": h_body}


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    import os, random, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def expect(cond, msg):
        if not cond:
            raise AssertionError(msg)

    # frame packing: serpentine remap + GRB order
    frame = [(r * 16, g * 16, 7) for r in range(4) for g in range(4)]
    raw = pack_frame(frame, 4, 4, serpentine=True)
    expect(len(raw) == 4 * 4 * 3, "pack_frame length")
    # logical (row=1, col=0) → physical index 1*4+3 = 7 (serpentine)
    r, g, b = frame[4]
    expect(raw[7 * 3] == g and raw[7 * 3 + 1] == r and raw[7 * 3 + 2] == b,
           "serpentine GRB remap")
    raw_ns = pack_frame(frame, 4, 4, serpentine=False)
    expect(raw_ns[4 * 3] == g and raw_ns[4 * 3 + 1] == r, "row-major remap")
    expect(pack_frames([frame, frame], 4, 4) == raw + raw, "pack_frames")
    print("PASS pack_frame/pack_frames")

    # py_sylvester + py_verify against the package reference
    from hoa64 import hadamard  # lazy
    for n in (1, 2, 4, 16, 64):
        Href = hadamard.sylvester(n)
        Hpy = py_sylvester(n)
        expect([[int(v) for v in row] for row in Href] == Hpy,
               f"py_sylvester({n}) != hadamard.sylvester")
        expect(py_verify(Hpy), f"py_verify({n}) failed")
        expect(hadamard.verify(Href), f"reference verify({n}) failed")
    expect(not py_verify(py_perturb(py_sylvester(8), random.Random(1))),
           "single flip should break orthogonality")
    print("PASS py_sylvester/py_verify vs hoa64.hadamard")

    # ils_step descends on a perturbed matrix
    rng = random.Random(42)
    H = py_sylvester(16)
    for _ in range(4):
        H = py_perturb(H, rng)
    H, improved, e0, e1 = py_ils_step(H)
    expect(e0 > 0 and improved and e1 < e0, "py_ils_step did not descend")
    print("PASS py_ils_step")

    # py_flux_map against micromag.flux_map on Sylvester-16
    import numpy as np
    from hoa64 import micromag  # lazy
    H16 = hadamard.sylvester(16)
    Wref = np.asarray(micromag.flux_map(H16), dtype=float)
    Wpy = np.asarray(py_flux_map([[int(v) for v in row] for row in H16]))
    expect(np.allclose(Wref, Wpy), "py_flux_map != micromag.flux_map")
    hist = py_tile_hist([[int(v) for v in row] for row in H16], tile=8)
    expect(len(hist) == 4, f"H.16 should have 4 unique 8×8 tiles, got {len(hist)}")
    w4 = py_wall4([[int(v) for v in row] for row in H16])
    expect(all(0.0 <= v <= 1.0 for row in w4 for v in row), "wall4 range")
    print("PASS py_flux_map vs micromag (H.16, 4 tiles)")

    # py_fbm determinism + range
    pts = [(0.13, 0.71), (0.5, 0.5), (0.999, 0.001), (0.3, 0.3)]
    for u, v in pts:
        a = py_fbm(u, v, octaves=5, seed=7)
        b = py_fbm(u, v, octaves=5, seed=7)
        expect(a == b, "py_fbm not deterministic")
        expect(-1.0 <= a <= 1.0, f"py_fbm out of range: {a}")
    expect(py_fbm(0.13, 0.71, seed=7) != py_fbm(0.13, 0.71, seed=8),
           "seed ignored")
    print("PASS py_fbm determinism/range")

    # firmware generation
    for board in ("esp32", "teensy", "circuitpython"):
        files = led_firmware({"board": board, "w": 16, "h": 8})
        expect("README.md" in files, f"{board}: no README")
        if board == "circuitpython":
            expect("code.py" in files and "neopixel" in files["code.py"],
                   "circuitpython firmware")
        else:
            name = f"hoa64_matrix_{board}.ino"
            expect(name in files and "FastLED" in files[name],
                   f"{board} firmware")
            expect("#define WIDTH 16" in files[name], "config header")
    mf = mesh_firmware({"n_nodes": 4, "gateway_id": 0})
    expect("hoa64_mesh.ino" in mf and "esp_now" in mf["hoa64_mesh.ino"],
           "mesh firmware")
    print("PASS led_firmware/mesh_firmware")

    # engine export: non-empty + expected markers
    markers = {
        ("hadamard_core", "circuitpython"): ("def sylvester",),
        ("flux_map", "circuitpython"): ("def flux_map",),
        ("terrain_fbm", "circuitpython"): ("def fbm",),
        ("hadamard_core", "rust_no_std"): ("no_std", "pub fn sylvester"),
        ("flux_map", "rust_no_std"): ("no_std", "pub fn flux_map4"),
        ("terrain_fbm", "rust_no_std"): ("no_std", "libm"),
        ("hadamard_core", "c_baremetal"): ("h64_sylvester", ".h"),
        ("flux_map", "c_baremetal"): ("h64_flux_map4",),
        ("terrain_fbm", "c_baremetal"): ("h64_fbm",),
    }
    for engine in ENGINES:
        for target in TARGETS:
            files = export_engine(engine, target)
            expect(files and all(len(v) > 0 for v in files.values()),
                   f"{engine}/{target}: empty export")
            body = files.get(f"{engine}.py") or files.get("lib.rs") \
                or files.get(f"{engine}.c")
            expect(body and len(body) > 200, f"{engine}/{target}: thin body")
            joined = "\n".join(files)
            text = "\n".join(files.values())
            for m in markers[(engine, target)]:
                expect(m in joined or m in text,
                       f"{engine}/{target}: missing marker {m!r}")
    cargo = export_engine("terrain_fbm", "rust_no_std")["Cargo.toml"]
    expect('libm = "*"' in cargo, "fbm rust export needs libm")
    expect('libm' not in export_engine("flux_map", "rust_no_std")["Cargo.toml"],
           "flux_map rust export should not need libm")
    print("PASS export_engine (3 engines × 3 targets)")

    print("mcu selftest OK")
