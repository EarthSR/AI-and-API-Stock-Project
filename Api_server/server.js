require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
app.use(cors());
app.use(express.json());

let accessToken = null;
let tokenExpires = 0; // Timestamp ของเวลาหมดอายุ

// ฟังก์ชันขอ Token ใหม่
const fetchPowerBIToken = async () => {
  try {
    const response = await axios.post(
      `https://login.microsoftonline.com/${process.env.TENANT_ID}/oauth2/token`,
      new URLSearchParams({
        grant_type: "client_credentials",
        client_id: process.env.CLIENT_ID,
        client_secret: process.env.CLIENT_SECRET,
        resource: "https://analysis.windows.net/powerbi/api",
      }),
      { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
    );

    accessToken = response.data.access_token;
    tokenExpires = Date.now() + response.data.expires_in * 1000;
    console.log("✅ Power BI Access Token Updated!");
  } catch (error) {
    console.error("❌ Error fetching Power BI Token:", error.response?.data || error.message);
  }
};

// API ให้ React ดึง Token ไปใช้
app.get("/get-token", async (req, res) => {
  if (!accessToken || Date.now() >= tokenExpires) {
    await fetchPowerBIToken(); // รีเฟรช Token เมื่อหมดอายุ
  }
  res.json({ access_token: accessToken });
});

// เริ่มต้น Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  fetchPowerBIToken(); // โหลด Token ตอนเริ่มต้น Server
});
