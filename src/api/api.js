const API_URL = " http:// 192.168.100.7:8000"; // Update this to match your backend IP if needed

export const loginUser = async (username, password) => {
  const res = await fetch(`${API_URL}/api/token/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return await res.json();
};

export const registerUser = async (username, password, email) => {
  const res = await fetch(`${API_URL}/api/register/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, email }),
  });
  return await res.json();
};
