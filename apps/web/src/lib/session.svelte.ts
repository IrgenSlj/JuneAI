// One session per window: the token handed over in the URL fragment, a client
// that sends it, and what the daemon says about the vault.
import { type JuneClient, type VaultInfo, browserTokenEnvironment, createJuneClient, takeApiToken } from "@june/ui/api";

class Session {
  token: string | undefined = $state();
  vault: VaultInfo | undefined = $state();
  error: string | undefined = $state();
  readonly client: JuneClient = createJuneClient();

  async start(): Promise<void> {
    this.token = takeApiToken(browserTokenEnvironment());
    this.client.setToken(this.token);
    if (this.token === undefined) return;
    try {
      this.vault = await this.client.getVault();
    } catch (error) {
      this.error = error instanceof Error ? error.message : String(error);
    }
  }
}

export const session = new Session();
