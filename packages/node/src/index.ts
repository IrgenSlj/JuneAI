export { type RunResult, type Runner, macKeychain, platformSecretStore, runProgram } from "./keychain.js";
export { type FileLockOptions, LockTimeoutError, nodeFileLock } from "./lock.js";
export { nodeVaultFs } from "./vault-fs.js";
export { type IndexedThread, type SearchHit, type SyncReport, type ThreadIndex, makeSnippet, openThreadIndex, queryTerms } from "./thread-index.js";
export { type ExportDocument, ExportReadError, documentsIn, readExportPath } from "./export-files.js";
