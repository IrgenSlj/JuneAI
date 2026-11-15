import { secretStoreContract } from "../testing/secret-store-contract.js";
import { memorySecretStore } from "./memory-secrets.js";

secretStoreContract("memory", async () => memorySecretStore());
