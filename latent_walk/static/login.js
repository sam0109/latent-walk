const form = document.querySelector("#loginForm");
const password = document.querySelector("#password");
const error = document.querySelector("#loginError");
const button = form.querySelector("button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  error.textContent = "";

  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: password.value }),
    });
    const result = await response.json();
    if (!response.ok) {
      error.textContent = result.detail || "Access denied.";
      password.select();
      return;
    }
    location.replace("/");
  } catch {
    error.textContent = "The studio could not be reached.";
  } finally {
    button.disabled = false;
  }
});

