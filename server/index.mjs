import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { ApiError, generate, listModels } from "./providers.mjs";
import { tracingStatus, traceModel } from "./tracing.mjs";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".png": "image/png",
  ".woff2": "font/woff2",
};
function json(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(data));
}

export function createAppServer({
  middleware,
  dist = resolve(root, "dist"),
} = {}) {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      if (url.pathname.startsWith("/api/")) {
        if (!["localhost", "127.0.0.1", "[::1]"].includes(url.hostname))
          throw new ApiError("Local access only.", 403);
        if (req.headers.origin && req.headers.origin !== url.origin)
          throw new ApiError("Cross-origin requests are not allowed.", 403);
        if (req.method === "GET" && url.pathname === "/api/tracing/status")
          return json(res, 200, await tracingStatus());
        if (req.method === "GET" && url.pathname === "/api/models") {
          return json(
            res,
            200,
            await listModels(Object.fromEntries(url.searchParams)),
          );
        }
        if (
          req.method === "POST" &&
          ["/api/generate", "/api/tracing/run"].includes(url.pathname)
        ) {
          if (!req.headers["content-type"]?.startsWith("application/json"))
            throw new ApiError("Send JSON.", 415);
          const chunks = [];
          let size = 0;
          for await (const chunk of req) {
            size += chunk.length;
            if (size > 150_000)
              throw new ApiError("Request is too large.", 413);
            chunks.push(chunk);
          }
          let input;
          try {
            input = JSON.parse(Buffer.concat(chunks).toString());
          } catch {
            throw new ApiError("Request is not valid JSON.");
          }
          if (!input || typeof input !== "object" || Array.isArray(input))
            throw new ApiError("Send a JSON object.");
          const controller = new AbortController();
          res.once("close", () => {
            if (!res.writableEnded) controller.abort();
          });
          if (url.pathname === "/api/tracing/run") {
            const event = (value) => {
              if (!res.destroyed) {
                if (!res.headersSent)
                  res.writeHead(200, {
                    "Content-Type": "application/x-ndjson",
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                  });
                res.write(JSON.stringify(value) + "\n");
              }
            };
            try {
              await traceModel(input, {
                signal: controller.signal,
                onEvent: event,
              });
            } catch (error) {
              if (!res.headersSent) throw error;
              event({ type: "error", error: error.message });
            }
            if (!res.destroyed) res.end();
            return;
          }
          const result = await generate(input, controller.signal);
          if (!res.destroyed) json(res, 200, result);
          return;
        }
        throw new ApiError("API route not found.", 404);
      }
      if (middleware) return middleware(req, res);
      if (req.method !== "GET" && req.method !== "HEAD")
        throw new ApiError("Method not allowed.", 405);
      const path = resolve(dist, "." + decodeURIComponent(url.pathname));
      if (path !== dist && !path.startsWith(dist + sep))
        throw new ApiError("Not found.", 404);
      let file = path;
      try {
        if (!(await stat(file)).isFile()) file = resolve(dist, "index.html");
      } catch {
        file = resolve(dist, "index.html");
      }
      const data = await readFile(file);
      res.writeHead(200, {
        "Content-Type": mime[extname(file)] || "application/octet-stream",
        "X-Content-Type-Options": "nosniff",
      });
      res.end(req.method === "HEAD" ? undefined : data);
    } catch (error) {
      if (res.destroyed) return;
      json(res, error.status || 500, {
        error:
          error instanceof ApiError
            ? error.message
            : "The request could not be completed. Check that the app is built and the model server is running.",
      });
    }
  });
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const dev = process.argv.includes("--dev");
  const port = Number(process.env.PORT || 5173);
  const vite = dev
    ? await (
        await import("vite")
      ).createServer({ root, server: { middlewareMode: true }, appType: "spa" })
    : null;
  const server = createAppServer({ middleware: vite?.middlewares });
  server.listen(port, "127.0.0.1", () =>
    console.log(`\n  TokenScope is running at http://127.0.0.1:${port}\n`),
  );
  const shutdown = () => {
    server.close();
    vite?.close();
    process.exit(0);
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}
