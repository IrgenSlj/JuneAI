import { vaultFsContract } from "../testing/vault-fs-contract.js";
import { memoryVaultFs } from "./memory.js";

vaultFsContract("memory", async () => memoryVaultFs());
