// Reconnecting WebSocket helper (Phase 2 job-progress plumbing).
// connect("/ws/jobs/xyz", { open, message, close, error }) -> { send, close }.
// Reconnects with exponential backoff (0.5 s → 10 s) until close() is called.

export function connect(path, handlers = {}) {
  let ws = null;
  let closed = false;
  let delay = 500;

  function open() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}${path}`);
    ws.onopen = (e) => {
      delay = 500;
      handlers.open?.(e);
    };
    ws.onmessage = (e) => {
      let data = e.data;
      try {
        data = JSON.parse(e.data);
      } catch {
        /* leave raw */
      }
      handlers.message?.(data, e);
    };
    ws.onerror = (e) => handlers.error?.(e);
    ws.onclose = (e) => {
      handlers.close?.(e);
      if (!closed) setTimeout(open, (delay = Math.min(delay * 2, 10000)));
    };
  }

  open();

  return {
    send(d) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(typeof d === "string" ? d : JSON.stringify(d));
      }
    },
    close() {
      closed = true;
      ws?.close();
    },
  };
}
