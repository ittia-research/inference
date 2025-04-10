Embedding inferencing on TPU with PyTorch XLA.

Visitor: Why another inferencing server?
Project: There are a lot of good inferencing project outthere. The project is mainly for edge cases in which we can not find a good existing project yet, mainly embedding inferencing on TPU device. We have noticed vLLM supports TPU and embedding, but can not make it working. If you know any tools that works good, let us know.

## Features
- Supported devices: Google TPU
- Supported models: Alibaba-NLP/gte-Qwen2-1.5B-instruct
- API: REST compateble with OpenAI format

## Quick Start
```bash
git pull https://github.com/ittia-research/inference
cd inference
docker compose up -d
docker compose logs -f
```

## Other Docs
[Roadmap](./docs/roadmap.md)

## Acknowledgements
- TPU Research Cloud team at Google
