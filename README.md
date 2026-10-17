# TokenScope

**See what your local language model chose—and what it almost said.**

TokenScope is a local web app for exploring the candidate tokens and probabilities behind a generated response. Connect Ollama or llama.cpp, ask a question, then hover over or click any token to inspect the alternatives recorded during generation.

![TokenScope showing a local model response, token probabilities, and an interactive probability chart](docs/screenshots/explorer.png)

## Features

- **Hover previews:** see candidate words and their probabilities beside each token, scroll the full returned list, or pin it to the inspector.
- **Token inspector:** ranked alternatives, candidate search, previous/next navigation, and probability mass outside the returned candidates.
- **Text and token views:** read the original response or inspect exact token boundaries, including spaces, punctuation, and word fragments.
- **Probability visualization:** optional token colors and an interactive chart of the chosen token's probability across the response.
- **Local model discovery:** choose an installed model and configure temperature, output length, candidate count, and random seed.
- **Exploration history and export:** revisit recent runs in the same browser and download the complete record as JSON.
- **Responsive interface:** desktop and mobile layouts, keyboard focus previews, and Escape to dismiss previews.
- **Local operation:** inference goes through a local server, and fonts are bundled with the app.

![Hover preview showing candidate tokens and their probabilities](docs/screenshots/hover-preview.png)

## Quick start

### Requirements

- Node.js **20.19+** or **22.12+**, including newer supported releases, and npm.
- [Ollama](https://ollama.com/) with a downloaded text-generation model, or a recent [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
- A model and runner that return token probability metadata.

### Install and run

```sh
git clone https://github.com/Amir-Parsian/llmMonitor.git
cd llmMonitor
npm ci
npm run dev
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)**.

The app discovers Ollama models at `http://127.0.0.1:11434`. If Ollama is not running, open the Ollama app or run this in another terminal:

```sh
ollama serve
```

If you need a model, download one, for example:

```sh
ollama pull gemma4:e4B
```

The built-in **Illustrative demo** works before a model is connected. Its probabilities and tokenization are synthetic and labeled accordingly.

### Production build

```sh
npm run build
npm start
```

The production app also runs at `http://127.0.0.1:5173`. To choose another port:

```sh
PORT=5174 npm start
```

The app binds to `127.0.0.1`. The UI and backend share an origin, so no Ollama CORS changes are needed.

## How to explore

1. Write a prompt and choose a local model.
2. Adjust temperature if desired, then select **Generate**. **Cmd/Ctrl + Enter** also generates.
3. Hover over a token for a scrollable candidate preview. Click it, or select **Pin to inspector**, to inspect it in the side panel.
4. Switch between **Text** and **Tokens**, toggle probability colors, or select a position in the probability chart.
5. Search candidates or move between tokens with the inspector arrows.
6. Select **Export exploration** to download the prompt, response, generation settings, and recorded probabilities as JSON.

Open **Connection settings** to change the server, requested candidate count, maximum output length, or seed. A fixed seed can help compare runs; exact repeatability depends on the model server.

Each generation is independent. Previous explorations are not sent as chat history. Recent explorations and settings are stored in this browser's local storage, up to 12 runs within the browser's storage capacity. Use Export to keep larger or important records.

## Model servers and candidate limits

| Provider  | Default address          | Candidates per token  | Integration                                               |
| --------- | ------------------------ | --------------------- | --------------------------------------------------------- |
| Ollama    | `http://127.0.0.1:11434` | Request up to **20**  | Native `/api/chat` with `logprobs` and `top_logprobs`     |
| llama.cpp | `http://127.0.0.1:8080`  | Request up to **100** | Native `/apply-template` and `/completion` with `n_probs` |

### Ollama

TokenScope requests `logprobs: true`, `stream: false`, `think: false`, and the selected `top_logprobs` count. Ollama currently limits `top_logprobs` to 20. Support depends on the installed version and model runner.

If the server does not return probabilities or alternatives, the app explains what is missing and preserves the response. It does not invent probabilities.

Real generation has been verified with **Ollama 0.40.1 and `gemma4:e4B`**.

### llama.cpp

Run a recent llama.cpp server with a compatible GGUF model:

```sh
llama-server -m /absolute/path/to/model.gguf --port 8080
```

Choose **llama.cpp** in Connection settings. TokenScope discovers models through `/v1/models`, applies the model's chat template through `/apply-template`, and requests probabilities through native `/completion` with `n_probs` and `post_sampling_probs: false`.

The adapter accepts both `probs` and `completion_probabilities` response fields. Servers may return fewer candidates than requested. A chosen token outside the shortlist is retained, so the displayed list may contain one additional selected token.

The llama.cpp integration is covered by contract tests against a simulated server. A live llama-server model run has not been verified in this project.

You may be able to use a compatible GGUF blob already downloaded by Ollama. Compatibility depends on your llama.cpp build. TokenScope does not download, convert, duplicate, or load model weights itself.

## Understanding the probabilities

Language models generate **tokens**, which may be whole words, word fragments, spaces, punctuation, or byte fragments. At each position, the model scores possible next tokens based on the prompt and previously generated tokens. Sampling settings influence which token is chosen; the most likely candidate does not always win.

TokenScope displays `exp(logprob)` as reported by the provider. It preserves the original probabilities and does **not** rescale the visible shortlist to sum to 100%. The inspector shows probability mass outside the returned list.

Provider-reported probabilities can differ from the final sampling distribution, depending on the server and sampler chain. These numbers express token likelihood, **not factual accuracy or confidence that an answer is correct**. The mean and chart summarize the reported probabilities of the chosen tokens.

Probabilities must be recorded **during generation**. Copying a response from another chat cannot recover its original candidate distributions.

## Privacy and storage

- Prompts travel from the browser through the local Node server to the selected loopback model endpoint.
- The backend accepts only local model server URLs and rejects cross-origin API requests.
- The app requires no API key or hosted service. With a downloaded model and local inference server, it runs locally; the interface makes no external font requests.
- Settings and recent explorations remain in browser local storage until cleared. Exports are saved only when requested.
- Connecting to an Ollama endpoint does not by itself make a cloud-backed model local. Choose a downloaded model for local inference.

## Development

Built with React, TypeScript, Vite, Lucide icons, and Node's native HTTP server.

| Command         | Purpose                                       |
| --------------- | --------------------------------------------- |
| `npm ci`        | Install the locked dependency versions        |
| `npm run dev`   | Run the local API and Vite interface          |
| `npm run build` | Type-check and build the production interface |
| `npm start`     | Serve the production build and local API      |
| `npm test`      | Run the backend tests                         |

```text
src/
  App.tsx                 Main interface and exploration state
  TokenHoverPreview.tsx   Hover and keyboard candidate previews
  demo.ts                 Labeled illustrative example
  format.ts               Shared probability and token formatting
  styles.css              Responsive interface styles
  types.ts                Shared frontend data types
server/
  index.mjs               Local HTTP API and frontend server
  providers.mjs           Ollama and llama.cpp integrations
  providers.test.mjs      Provider, probability, and API tests
public/
  fonts/                  Bundled fonts and their licenses
docs/screenshots/         Interface screenshots
```

The backend validates generation controls, normalizes provider responses, preserves UTF-8 byte offsets, and aborts inference requests when the client disconnects. The response is delivered after generation completes, together with the recorded token metadata. **Stop** cancels the current run and discards late results.

The 10 backend tests cover probability conversion, shortlist mass, selected tokens outside the shortlist, UTF-8 fragments, reasoning-token alignment, provider requests, 100-candidate llama.cpp responses, model discovery, missing metadata, and API validation. Browser verification covers hovering, scrolling candidates, pinning, search, display modes, keyboard dismissal, cancellation, export, and responsive layout.

## Troubleshooting

| Issue                             | What to check                                                                                                   |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Cannot connect to Ollama          | Open Ollama or run `ollama serve`, check the URL in settings, then reconnect.                                   |
| No text models are listed         | Download a text-generation model and reconnect. Image-only models are excluded where capabilities are reported. |
| Response has no probability data  | Try a newer model server or another model/runner. Text alone cannot supply the missing data.                    |
| Only 20 candidates are available  | This is Ollama's current API limit. Connect llama.cpp to request up to 100.                                     |
| Response ends mid-sentence        | Increase **Maximum output tokens**. The app reports when the output limit is reached.                           |
| Port 5173 is already in use       | Stop the other instance or set `PORT`, for example `PORT=5174 npm run dev`.                                     |
| Interface changes are not visible | Rebuild the production app and refresh the browser. Vite updates interface files automatically in development.  |

## License

TokenScope's original source code is available under the [MIT License](LICENSE), copyright © 2026 Amir Parsian. Bundled fonts retain their [SIL Open Font License notices](public/fonts/); dependencies and model weights remain subject to their respective licenses.

## References

- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama request types and candidate limits](https://github.com/ollama/ollama/blob/main/api/types.go)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Hugging Face text generation guide](https://huggingface.co/docs/transformers/main/en/llm_tutorial)
