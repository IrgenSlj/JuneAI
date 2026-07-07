import { test, expect } from "@playwright/test";
import { MOCK_API_BASE, mockApi } from "./_mocks.js";

test("Chat page offers conversational initiative and sends the selected prompt", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));

  await mockApi(page);

  let sentMessage = "";
  await page.route(`${MOCK_API_BASE}/chat`, async (route) => {
    const body = route.request().postDataJSON() as { message?: string };
    sentMessage = body.message ?? "";
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body:
        [
          {
            type: "token",
            content: "Let's start with the next hour.",
            tool_name: "",
            tool_result: "",
          },
          { type: "done", content: "", tool_name: "", tool_result: "" },
        ]
          .map((event) => `data: ${JSON.stringify(event)}`)
          .join("\n\n") + "\n\n",
    });
  });

  await page.goto("/chat");

  await expect(page.getByText("I could start here")).toBeVisible();
  await page
    .getByRole("button", { name: /Want to shape the next hour\?/ })
    .click();

  await expect(page.getByText("Let's start with the next hour.")).toBeVisible();
  expect(sentMessage).toBe("Help me plan the next hour.");
  expect(pageErrors, `Uncaught page errors: ${pageErrors.join(" | ")}`).toHaveLength(
    0,
  );
});
