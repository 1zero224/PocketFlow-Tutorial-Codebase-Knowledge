import { chunkBatch } from 'code-chunk'

const DEFAULT_OPTIONS = {
  maxChunkSize: 3000,
  contextMode: 'full',
  siblingDetail: 'signatures',
  overlapLines: 5,
  concurrency: 8,
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let input = ''
    process.stdin.setEncoding('utf8')
    process.stdin.on('data', (chunk) => {
      input += chunk
    })
    process.stdin.on('end', () => resolve(input))
    process.stdin.on('error', reject)
  })
}

function validatePayload(payload) {
  if (!payload || !Array.isArray(payload.files)) {
    throw new Error('Input JSON must contain a files array')
  }
  for (const file of payload.files) {
    if (!Number.isInteger(file.file_index)) {
      throw new Error('Each file must contain an integer file_index')
    }
    if (typeof file.filepath !== 'string') {
      throw new Error('Each file must contain a string filepath')
    }
    if (typeof file.code !== 'string') {
      throw new Error('Each file must contain a string code')
    }
  }
}

function entityName(entity) {
  if (!entity) {
    return ''
  }
  const scope = Array.isArray(entity.scope)
    ? entity.scope.map((item) => item?.name).filter(Boolean)
    : []
  return [...scope, entity.name].filter(Boolean).join(' > ')
}

function mapChunk(chunk) {
  const context = chunk.context ?? {}
  const entities = Array.isArray(context.entities) ? context.entities : []
  const primary =
    entities.find((entity) => !['import', 'export'].includes(entity?.type)) ??
    entities[0] ??
    null
  const imports = Array.isArray(context.imports) ? context.imports : []

  return {
    index: chunk.index,
    content: chunk.text ?? '',
    context_text: chunk.contextualizedText ?? '',
    line_range: chunk.lineRange ?? null,
    language: context.language ?? '',
    chunk_kind: primary ? 'entity' : 'misc',
    symbol_path: entityName(primary) || '<module>',
    signature: primary?.signature ?? '',
    parent_scope: Array.isArray(context.scope)
      ? context.scope.map((item) => item?.name).filter(Boolean).join(' > ')
      : '',
    related_imports: imports
      .map((item) => item?.source || item?.name)
      .filter(Boolean),
  }
}

function mapResult(result, inputFile) {
  if (result?.error) {
    return {
      file_index: inputFile.file_index,
      filepath: inputFile.filepath,
      error: result.error.message ?? String(result.error),
    }
  }

  const chunks = Array.isArray(result?.chunks) ? result.chunks.map(mapChunk) : []
  return {
    file_index: inputFile.file_index,
    filepath: inputFile.filepath,
    chunks,
  }
}

async function main() {
  try {
    const raw = await readStdin()
    const payload = JSON.parse(raw || '{}')
    validatePayload(payload)

    const files = payload.files.map((file) => ({
      filepath: file.filepath,
      code: file.code,
    }))
    const options = { ...DEFAULT_OPTIONS, ...(payload.options ?? {}) }
    const batchResults = await chunkBatch(files, options)
    const results = payload.files.map((file, index) =>
      mapResult(batchResults[index], file),
    )

    process.stdout.write(JSON.stringify({ results }))
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        error: error?.message ?? String(error),
      }),
    )
    process.exitCode = 1
  }
}

await main()
