const express = require("express");
const bodyParser = require("body-parser");
const mysql = require("mysql2");
const mysqlpromise = require("mysql2/promise");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const admin = require("firebase-admin");
const cors = require("cors");
const axios = require("axios");
const fs = require("fs");
const crypto = require("crypto");
const nodemailer = require("nodemailer");
const multer = require("multer");
require("dotenv").config();
const path = require("path");
const cron = require("node-cron");
const JWT_SECRET = process.env.JWT_SECRET;
const app = express();
const { PythonShell } = require("python-shell");
const serviceAccount = require("./config/trademine-a3921-firebase-adminsdk-fbsvc-ff0de5bd4d.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors()); // Enable CORS
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: process.env.DB_PORT,
  waitForConnections: true,
  connectionLimit: 20,
  queueLimit: 0,
  connectTimeout: 60000,
});

const pool_notification = mysqlpromise.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: process.env.DB_PORT,
  waitForConnections: true,
  connectionLimit: 20,
  queueLimit: 0,
  connectTimeout: 60000,
});

// ฟังก์ชันสำหรับตรวจสอบ JWT token
const verifyToken = (req, res, next) => {
  const token = req.headers["authorization"];

  if (!token) {
    return res.status(403).json({ message: "Token is required" });
  }

  // ตัดคำว่า "Bearer" ออก
  const bearerToken = token.split(" ")[1];

  jwt.verify(bearerToken, JWT_SECRET, (err, decoded) => {
    if (err) {
      return res.status(401).json({ message: "Invalid token" });
    }

    req.userId = decoded.id; // ✅ เก็บ userId
    req.role = decoded.role; // ✅ เก็บ role ไว้ใช้ต่อใน verifyAdmin
    next();
  });
};

module.exports = verifyToken;

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "./uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});
const upload = multer({ storage: storage });

function generateOtp() {
  const otp = crypto.randomBytes(3).toString("hex"); // 3 bytes = 6 hex characters
  return parseInt(otp, 16).toString().slice(0, 6); // Convert hex to decimal and take first 6 digits
}

function sendOtpEmail(email, otp, callback) {
  const transporter = nodemailer.createTransport({
    service: "Gmail",
    auth: {
      user: process.env.email,
      pass: process.env.emailpassword,
    },
  });

  const mailOptions = {
    from: process.env.email,
    to: email,
    subject: "Your OTP Code",
    html: `
        <div style="font-family: Arial, sans-serif; color: #333;">
          <h2 style="color: #007bff;">Your OTP Code</h2>
          <p>Hello,</p>
          <p>We received a request to verify your email address. Please use the OTP code below to complete the process:</p>
          <div style="padding: 10px; border: 2px solid #007bff; display: inline-block; font-size: 24px; color: #007bff; font-weight: bold;">
            ${otp}
          </div>
          <p>This code will expire in 10 minutes.</p>
          <p>If you didnt request this, please ignore this email.</p>
          <p style="margin-top: 20px;">Thanks, <br> The Team</p>
          <hr>
          <p style="font-size: 12px; color: #999;">This is an automated email, please do not reply.</p>
        </div>
      `,
  };

  transporter.sendMail(mailOptions, (error, info) => {
    if (error) {
      console.error("Error sending OTP email:", error); // Log the error for debugging purposes
      return callback({
        error: "Failed to send OTP email. Please try again later.",
      });
    }
    callback(null, info); // Proceed if the email was successfully sent
  });
}

// User-Register-Email
app.post("/api/register/email", async (req, res) => {
  try {
    const { email, role = "user" } = req.body; // ถ้า Role ไม่ได้ส่งมา ให้ Default เป็น "user"

    if (!email) {
      return res.status(400).json({ error: "กรุณากรอกอีเมล" });
    }

    pool.query(
      "SELECT * FROM User WHERE Email = ?",
      [email],
      (err, results) => {
        if (err) {
          console.error("Database error during email check:", err);
          return res
            .status(500)
            .json({ error: "Database error during email check" });
        }

        if (results.length > 0) {
          const user = results[0];

          // ถ้า Email นี้เคยลงทะเบียนแล้วและเป็น Active
          if (user.Status === "active" && user.Password) {
            return res.status(400).json({ error: "Email Already use" });
          }

          // ถ้าเคยสมัครแต่เป็น deactivated ให้เปิดใช้งานอีกครั้ง
          if (user.Status === "deactivated") {
            pool.query("UPDATE User SET Status = 'active' WHERE Email = ?", [
              email,
            ]);
            return res
              .status(200)
              .json({ message: "บัญชีของคุณถูกเปิดใช้งานอีกครั้ง" });
          }
        }

        // **สร้าง OTP และกำหนดเวลา Expiry**
        const otp = generateOtp();
        const expiresAt = new Date(Date.now() + 3 * 60 * 1000); // OTP หมดอายุใน 3 นาที
        const createdAt = new Date(Date.now());

        // **เพิ่มข้อมูล User ใหม่หากยังไม่มี**
        pool.query(
          `INSERT INTO User (Email, Username, Password, Role, Status) 
              VALUES (?, '', '', ?, 'pending') 
              ON DUPLICATE KEY UPDATE Status = 'pending', Role = ?`, // ✅ เพิ่ม Role
          [email, role, role], // ✅ กำหนดค่า Role
          (err) => {
            if (err) {
              console.error(
                "Database error during User insertion or update:",
                err
              );
              return res.status(500).json({
                error: "Database error during User insertion or update",
              });
            }

            // **ดึง UserID จาก Email**
            pool.query(
              "SELECT UserID FROM User WHERE Email = ?",
              [email],
              (err, results) => {
                if (err) {
                  console.error("Error fetching UserID:", err);
                  return res
                    .status(500)
                    .json({ error: "Database error fetching UserID" });
                }

                if (results.length === 0) {
                  return res.status(404).json({ error: "User not found" });
                }

                const userId = results[0].UserID;

                // **แทรก OTP โดยใช้ UserID แทน Email**
                pool.query(
                  `INSERT INTO OTP (OTP_Code, Created_At, Expires_At, UserID) 
                              VALUES (?, ?, ?, ?) 
                              ON DUPLICATE KEY UPDATE OTP_Code = ?, Created_At = ?, Expires_At = ?`,
                  [
                    otp,
                    createdAt,
                    expiresAt,
                    userId,
                    otp,
                    createdAt,
                    expiresAt,
                  ],
                  (err) => {
                    if (err) {
                      console.error("Error during OTP insertion:", err);
                      return res
                        .status(500)
                        .json({ error: "Database error during OTP insertion" });
                    }

                    console.log("OTP inserted successfully");
                    sendOtpEmail(email, otp, (error) => {
                      if (error)
                        return res
                          .status(500)
                          .json({ error: "Error sending OTP email" });
                      res
                        .status(200)
                        .json({ message: "OTP ถูกส่งไปยังอีเมลของคุณ" });
                    });
                  }
                );
              }
            );
          }
        );
      }
    );
  } catch (error) {
    console.error("Internal server error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Verify-OTP
app.post("/api/register/verify-otp", async (req, res) => {
  try {
    const { email, otp } = req.body;

    // ตรวจสอบว่า Email กับ OTP ถูกส่งมาหรือไม่
    if (!email || !otp) {
      return res.status(400).json({ error: "Email หรือ OTP ไม่ถูกต้อง" });
    }

    // ค้นหา UserID จาก Email
    pool.query(
      "SELECT UserID FROM User WHERE Email = ?",
      [email],
      (err, userResults) => {
        if (err) return res.status(500).json({ error: "Database error" });

        if (userResults.length === 0) {
          return res.status(404).json({ error: "ไม่พบ Email ในระบบ" });
        }

        const userId = userResults[0].UserID;

        // ค้นหา OTP ในฐานข้อมูลโดยใช้ UserID และ OTP
        pool.query(
          "SELECT * FROM OTP WHERE UserID = ? AND OTP_Code = ?",
          [userId, otp],
          (err, otpResults) => {
            if (err) return res.status(500).json({ error: "Database error" });

            // ถ้าไม่พบ OTP ในฐานข้อมูล
            if (otpResults.length === 0) {
              return res.status(400).json({ error: "Invalid OTP" });
            }

            // ตรวจสอบว่า OTP ยังไม่หมดอายุ
            const { Expires_At } = otpResults[0];
            if (new Date() > new Date(Expires_At)) {
              return res.status(400).json({ error: "Expired OTP" });
            }

            // ถ้า OTP ถูกต้องและไม่หมดอายุ
            res
              .status(200)
              .json({ message: "OTP is correct, you can set a password." });
          }
        );
      }
    );
  } catch (error) {
    res.status(500).json({ error: "Internal server error" });
  }
});

// User-Set-Password (อัปเดตใหม่)
app.post("/api/register/set-password", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Password is required" });
    }

    const hash = await bcrypt.hash(password, 10);

    // ดึง UserID ก่อนอัปเดตรหัสผ่าน
    pool.query(
      "SELECT UserID FROM User WHERE Email = ?",
      [email],
      (err, results) => {
        if (err) {
          console.error("Error fetching UserID:", err);
          return res
            .status(500)
            .json({ error: "Database error fetching UserID" });
        }

        if (results.length === 0) {
          return res
            .status(404)
            .json({ error: "No account found with this Email" });
        }

        const userId = results[0].UserID;

        // อัปเดตรหัสผ่านในตาราง User แต่ยังไม่เปลี่ยนเป็น 'active'
        pool.query(
          "UPDATE User SET Password = ?, Status = 'active' WHERE UserID = ?",
          [hash, userId],
          (err, results) => {
            if (err) {
              console.error("Database error during User update:", err);
              return res
                .status(500)
                .json({ error: "Database error during User update" });
            }

            if (results.affectedRows === 0) {
              return res
                .status(404)
                .json({ error: "Unable to update password" });
            }

            // ลบ OTP ที่เกี่ยวข้องกับ UserID
            pool.query("DELETE FROM OTP WHERE UserID = ?", [userId], (err) => {
              if (err) {
                console.error("Error during OTP deletion:", err);
                return res
                  .status(500)
                  .json({ error: "Error during OTP deletion" });
              }

              // **สร้าง Token และส่งกลับ**
              const token = jwt.sign({ id: userId }, JWT_SECRET, {
                expiresIn: "7d",
              });

              res.status(200).json({
                message:
                  "Password has been set successfully. Please complete your profile.",
                token: token,
              });
            });
          }
        );
      }
    );
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Resend-OTP
app.post("/api/resend-otp/register", async (req, res) => {
  try {
    const { email } = req.body; // ใช้ Email แทน UserID

    // ตรวจสอบว่า Email ถูกส่งมาหรือไม่
    if (!email) return res.status(400).json({ error: "กรุณากรอกอีเมล" });

    const newOtp = generateOtp();
    const newExpiresAt = new Date(Date.now() + 10 * 60 * 1000); // OTP หมดอายุใน 10 นาที

    // ค้นหา Email ในตาราง User เพื่อดึง UserID
    pool.query(
      "SELECT UserID FROM User WHERE Email = ?",
      [email],
      (err, userResults) => {
        if (err)
          return res
            .status(500)
            .json({ error: "Database error during user lookup" });
        if (userResults.length === 0)
          return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });

        const userId = userResults[0].UserID; // ดึง UserID ที่แท้จริง

        // ค้นหา OTP ที่มีอยู่ ถ้ามีให้อัปเดต ถ้าไม่มีให้แทรกใหม่
        pool.query(
          "SELECT * FROM OTP WHERE UserID = ?",
          [userId],
          (err, otpResults) => {
            if (err)
              return res
                .status(500)
                .json({ error: "Database error during OTP check" });

            if (otpResults.length > 0) {
              // ถ้ามี OTP อยู่แล้ว อัปเดตข้อมูลใหม่
              pool.query(
                "UPDATE OTP SET OTP_Code = ?, Expires_At = ? WHERE UserID = ?",
                [newOtp, newExpiresAt, userId],
                (err) => {
                  if (err)
                    return res
                      .status(500)
                      .json({ error: "Database error during OTP update" });

                  // ส่ง OTP ไปยังอีเมลของผู้ใช้
                  sendOtpEmail(email, newOtp, (error) => {
                    if (error)
                      return res
                        .status(500)
                        .json({ error: "Error sending OTP email" });
                    res.status(200).json({ message: "OTP ถูกส่งใหม่แล้ว" });
                  });
                }
              );
            } else {
              // ถ้าไม่มี OTP อยู่ก่อน ให้แทรกใหม่
              pool.query(
                "INSERT INTO OTP (UserID, OTP_Code, Created_At, Expires_At) VALUES (?, ?, NOW(), ?)",
                [userId, newOtp, newExpiresAt],
                (err) => {
                  if (err)
                    return res
                      .status(500)
                      .json({ error: "Database error during OTP insertion" });

                  // ส่ง OTP ไปยังอีเมลของผู้ใช้
                  sendOtpEmail(email, newOtp, (error) => {
                    if (error)
                      return res
                        .status(500)
                        .json({ error: "Error sending OTP email" });
                    res.status(200).json({ message: "OTP ถูกส่งใหม่แล้ว" });
                  });
                }
              );
            }
          }
        );
      }
    );
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Forgot-Passord
app.post("/api/forgot-password", async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ error: "Email is required" });
    }

    // ตรวจสอบว่ามี Email นี้ในตาราง User และบัญชีต้องเป็น active
    const userCheckSql =
      "SELECT UserID FROM User WHERE Email = ? AND Password IS NOT NULL AND Status = 'active'";
    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err)
        return res
          .status(500)
          .json({ error: "Database error during email check", details: err });

      if (userResults.length === 0) {
        return res.status(400).json({ error: "Email not found or inactive" });
      }

      const userId = userResults[0].UserID; // ดึง UserID จาก Email

      // สร้าง OTP ใหม่
      const otp = generateOtp();
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // OTP หมดอายุใน 10 นาที

      // ตรวจสอบว่า User มี OTP อยู่แล้วหรือไม่
      pool.query(
        "SELECT * FROM OTP WHERE UserID = ?",
        [userId],
        (err, otpResults) => {
          if (err)
            return res
              .status(500)
              .json({ error: "Database error during OTP check", details: err });

          if (otpResults.length > 0) {
            // ถ้ามี OTP อยู่แล้ว ให้ทำการอัปเดต
            const updateOtpSql =
              "UPDATE OTP SET OTP_Code = ?, Expires_At = ?, Created_At = NOW() WHERE UserID = ?";
            pool.query(updateOtpSql, [otp, expiresAt, userId], (err) => {
              if (err)
                return res.status(500).json({
                  error: "Database error during OTP update",
                  details: err,
                });

              sendOtpEmail(email, otp, (error) => {
                if (error)
                  return res
                    .status(500)
                    .json({ error: "Error sending OTP email" });
                res.status(200).json({ message: "OTP sent to email" });
              });
            });
          } else {
            // ถ้ายังไม่มี OTP ให้เพิ่มเข้าไป
            const saveOtpSql =
              "INSERT INTO OTP (UserID, OTP_Code, Expires_At, Created_At) VALUES (?, ?, ?, NOW())";
            pool.query(saveOtpSql, [userId, otp, expiresAt], (err) => {
              if (err)
                return res.status(500).json({
                  error: "Database error during OTP save",
                  details: err,
                });

              sendOtpEmail(email, otp, (error) => {
                if (error)
                  return res
                    .status(500)
                    .json({ error: "Error sending OTP email" });
                res.status(200).json({ message: "OTP sent to email" });
              });
            });
          }
        }
      );
    });
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Verify Reset OTP
app.post("/api/verify-reset-otp", async (req, res) => {
  try {
    const { email, otp } = req.body;

    if (!email || !otp) {
      return res.status(400).json({ error: "Email และ OTP ต้องถูกต้อง" });
    }

    // ค้นหา UserID จาก Email ก่อน
    pool.query(
      "SELECT UserID FROM User WHERE Email = ?",
      [email],
      (err, userResults) => {
        if (err)
          return res
            .status(500)
            .json({ error: "Database error during user lookup" });
        if (userResults.length === 0)
          return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });

        const userId = userResults[0].UserID;

        // ค้นหา OTP ในตาราง OTP โดยใช้ UserID และ OTP ที่ส่งมา
        pool.query(
          "SELECT OTP_Code, Expires_At FROM OTP WHERE UserID = ? AND OTP_Code = ?",
          [userId, otp],
          (err, results) => {
            if (err)
              return res
                .status(500)
                .json({ error: "Database error during OTP verification" });

            // ถ้าไม่พบ OTP ในฐานข้อมูล
            if (results.length === 0) {
              return res.status(400).json({ error: "OTP ไม่ถูกต้อง" });
            }

            const { Expires_At } = results[0];
            const now = new Date();

            // ตรวจสอบว่า OTP หมดอายุหรือไม่
            if (now > new Date(Expires_At)) {
              return res.status(400).json({ error: "OTP หมดอายุ" });
            }

            // ถ้า OTP ถูกต้องและยังไม่หมดอายุ
            res
              .status(200)
              .json({ message: "OTP ถูกต้อง คุณสามารถตั้งรหัสผ่านใหม่ได้" });
          }
        );
      }
    );
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Reset Password
app.post("/api/reset-password", async (req, res) => {
  try {
    const { email, newPassword } = req.body;

    if (!email || !newPassword) {
      return res.status(400).json({ error: "Email และ Password ต้องถูกต้อง" });
    }

    const hashedPassword = await bcrypt.hash(newPassword, 10);

    // ค้นหา UserID จาก Email ก่อน
    pool.query(
      "SELECT UserID FROM User WHERE Email = ?",
      [email],
      (err, userResults) => {
        if (err)
          return res
            .status(500)
            .json({ error: "Database error during user lookup" });
        if (userResults.length === 0)
          return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });

        const userId = userResults[0].UserID;

        // อัปเดตรหัสผ่านในตาราง User
        pool.query(
          "UPDATE User SET Password = ?, Status = 'active' WHERE Email = ?",
          [hashedPassword, email],
          (err) => {
            if (err) {
              console.error("Database error during password update:", err);
              return res
                .status(500)
                .json({ error: "Database error during password update" });
            }

            // ลบ OTP ที่เกี่ยวข้องกับ UserID จากตาราง OTP
            pool.query("DELETE FROM OTP WHERE UserID = ?", [userId], (err) => {
              if (err) {
                console.error("Error during OTP deletion:", err);
                return res
                  .status(500)
                  .json({ error: "Error during OTP deletion" });
              }

              res
                .status(200)
                .json({ message: "รหัสผ่านถูกตั้งใหม่เรียบร้อยแล้ว" });
            });
          }
        );
      }
    );
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Resend OTP for Reset Password
app.post("/api/resend-otp/reset-password", async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ error: "Please Enter You Email" });
    }

    // ตรวจสอบว่ามี Email นี้ในตาราง User
    const userCheckSql = "SELECT UserID FROM User WHERE Email = ?";
    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err)
        return res
          .status(500)
          .json({ error: "Database error during user lookup" });
      if (userResults.length === 0)
        return res.status(404).json({ error: "User not found" });

      const userId = userResults[0].UserID; // ดึง UserID จาก Email

      // สร้าง OTP ใหม่
      const otp = generateOtp();
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // OTP หมดอายุใน 10 นาที

      // ตรวจสอบว่า User มี OTP อยู่แล้วหรือไม่
      pool.query(
        "SELECT * FROM OTP WHERE UserID = ?",
        [userId],
        (err, otpResults) => {
          if (err)
            return res
              .status(500)
              .json({ error: "Database error during OTP lookup" });

          if (otpResults.length > 0) {
            // ถ้ามี OTP อยู่แล้ว ให้ทำการอัปเดต
            const updateOtpSql =
              "UPDATE OTP SET OTP_Code = ?, Expires_At = ? WHERE UserID = ?";
            pool.query(updateOtpSql, [otp, expiresAt, userId], (err) => {
              if (err)
                return res
                  .status(500)
                  .json({ error: "Database error during OTP update" });

              sendOtpEmail(email, otp, (error) => {
                if (error)
                  return res
                    .status(500)
                    .json({ error: "Error sending OTP email" });
                res.status(200).json({ message: "New OTP sent to email" });
              });
            });
          } else {
            // ถ้ายังไม่มี OTP ให้เพิ่มเข้าไป
            const insertOtpSql =
              "INSERT INTO OTP (UserID, OTP_Code, Created_At, Expires_At) VALUES (?, ?, NOW(), ?)";
            pool.query(insertOtpSql, [userId, otp, expiresAt], (err) => {
              if (err)
                return res
                  .status(500)
                  .json({ error: "Database error during OTP insert" });

              sendOtpEmail(email, otp, (error) => {
                if (error)
                  return res
                    .status(500)
                    .json({ error: "Error sending OTP email" });
                res.status(200).json({ message: "New OTP sent to email" });
              });
            });
          }
        }
      );
    });
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Login
app.post("/api/login", async (req, res) => {
  let conn;
  try {
    const { email, password, googleId } = req.body;
    const ipAddress =
      req.headers["x-forwarded-for"] || req.connection.remoteAddress;

    if (!email) {
      return res.status(400).json({ message: "กรุณากรอกอีเมล" });
    }

    conn = await pool.promise().getConnection();
    await conn.beginTransaction();

    const [rows] = await conn.query("SELECT * FROM User WHERE Email = ?", [
      email,
    ]);

    if (googleId) {
      if (rows.length === 0) {
        const [dupGid] = await conn.query(
          "SELECT UserID FROM User WHERE GoogleID = ? LIMIT 1",
          [googleId]
        );
        if (dupGid.length > 0) {
          await conn.rollback();
          return res
            .status(409)
            .json({ message: "บัญชี Google นี้ถูกใช้กับอีเมลอื่นแล้ว" });
        }

        const baseUsername =
          email
            .split("@")[0]
            .replace(/[^a-zA-Z0-9._-]/g, "")
            .slice(0, 20) || "user";
        let username = baseUsername;
        let suffix = 0;
        while (true) {
          const [u] = await conn.query(
            "SELECT 1 FROM User WHERE Username = ? LIMIT 1",
            [username]
          );
          if (u.length === 0) break;
          suffix += 1;
          username = `${baseUsername}${suffix}`;
        }

        const [ins] = await conn.query(
          `INSERT INTO User (Email, Username, GoogleID, Status, Role , LastLogin, LastLoginIP)
           VALUES (?, ?, ?, 'active', 'user', NOW(), ?)`,
          [email, username, googleId, ipAddress]
        );

        const newUserId = ins.insertId;

        const token = jwt.sign(
          { id: newUserId, email, role: "user", provider: "google", googleId },
          JWT_SECRET,
          { expiresIn: "7d" }
        );

        await conn.commit();
        return res.status(200).json({
          message: "สมัครและเข้าสู่ระบบด้วย Google สำเร็จ",
          token,
          user: {
            id: newUserId,
            email,
            username,
            role: "user",
          },
        });
      }

      const user = rows[0];

      if (user.Status !== "active") {
        await conn.rollback();
        return res.status(403).json({ message: "บัญชีถูกระงับการใช้งาน" });
      }

      if (user.GoogleID && user.GoogleID !== googleId) {
        await conn.rollback();
        return res
          .status(400)
          .json({ message: "บัญชีนี้ถูกผูกกับ Google คนละไอดี" });
      }

      if (!user.GoogleID) {
        const [dupGid2] = await conn.query(
          "SELECT UserID FROM User WHERE GoogleID = ? AND UserID <> ? LIMIT 1",
          [googleId, user.UserID]
        );
        if (dupGid2.length > 0) {
          await conn.rollback();
          return res
            .status(409)
            .json({ message: "บัญชี Google นี้ถูกใช้กับอีเมลอื่นแล้ว" });
        }

        await conn.query("UPDATE User SET GoogleID = ? WHERE UserID = ?", [
          googleId,
          user.UserID,
        ]);
      }

      await conn.query(
        "UPDATE User SET LastLogin = NOW(), LastLoginIP = ? WHERE UserID = ?",
        [ipAddress, user.UserID]
      );

      const token = jwt.sign(
        {
          id: user.UserID,
          email: user.Email,
          role: user.Role,
          provider: "google",
          googleId,
        },
        JWT_SECRET,
        { expiresIn: "7d" }
      );

      await conn.commit();
      return res.status(200).json({
        message: "เข้าสู่ระบบด้วย Google สำเร็จ",
        token,
        user: {
          id: user.UserID,
          email: user.Email,
          username: user.Username,
          role: user.Role,
        },
      });
    }

    if (rows.length === 0) {
      await conn.rollback();
      return res.status(404).json({ message: "ไม่พบบัญชีนี้" });
    }

    const user = rows[0];

    if (!password) {
      await conn.rollback();
      return res.status(400).json({ message: "กรุณากรอกรหัสผ่าน" });
    }

    if (user.Status !== "active") {
      await conn.rollback();
      return res.status(403).json({ message: "บัญชีถูกระงับการใช้งาน" });
    }

    if (user.FailedAttempts >= 5 && user.LastFailedAttempt) {
      const timeSinceLastAttempt =
        Date.now() - new Date(user.LastFailedAttempt).getTime();
      if (timeSinceLastAttempt < 300000) {
        await conn.rollback();
        return res.status(429).json({
          message: "คุณล็อกอินผิดพลาดหลายครั้ง โปรดลองอีกครั้งใน 5 นาที",
        });
      }
    }

    const isMatch = await bcrypt.compare(password, user.Password || "");
    if (!isMatch) {
      await conn.query(
        "UPDATE User SET FailedAttempts = FailedAttempts + 1, LastFailedAttempt = NOW() WHERE UserID = ?",
        [user.UserID]
      );
      await conn.commit();
      return res.status(401).json({ message: "อีเมลหรือรหัสผ่านไม่ถูกต้อง" });
    }

    await conn.query(
      "UPDATE User SET FailedAttempts = 0, LastLogin = NOW(), LastLoginIP = ? WHERE UserID = ?",
      [ipAddress, user.UserID]
    );

    const token = jwt.sign({ id: user.UserID, role: user.Role }, JWT_SECRET, {
      expiresIn: "7d",
    });

    await conn.commit();
    return res.status(200).json({
      message: "เข้าสู่ระบบสำเร็จ",
      token,
      user: {
        id: user.UserID,
        email: user.Email,
        username: user.Username,
        role: user.Role,
      },
    });
  } catch (error) {
    if (conn)
      try {
        await conn.rollback();
      } catch (_) {}
    console.error("Internal error:", error);
    res.status(500).json({ error: "Internal server error" });
  } finally {
    if (conn) conn.release();
  }
});

// Set Profile และ Login อัตโนมัติหลังจากตั้งโปรไฟล์เสร็จ
app.post(
  "/api/set-profile",
  verifyToken,
  upload.single("picture"),
  (req, res) => {
    const { newUsername, birthday, gender } = req.body;
    const userId = req.userId; // รับ UserID จาก token
    const picture = req.file ? `/uploads/${req.file.filename}` : null;

    // ตรวจสอบค่าที่ต้องมี
    if (!newUsername || !picture || !birthday || !gender) {
      return res.status(400).json({
        message: "New username, picture, birthday, and gender are required",
      });
    }

    // ตรวจสอบว่าค่า gender ถูกต้อง
    const validGenders = ["Male", "Female", "Other"];
    if (!validGenders.includes(gender)) {
      return res.status(400).json({
        message: "Invalid gender. Please choose 'Male', 'Female', or 'Other'.",
      });
    }

    // แปลงวันที่จาก DD/MM/YYYY เป็น YYYY-MM-DD
    const birthdayParts = birthday.split("/");
    const formattedBirthday = `${birthdayParts[2]}-${birthdayParts[1]}-${birthdayParts[0]}`;

    // อัปเดตโปรไฟล์ของผู้ใช้ และเปลี่ยนสถานะเป็น Active
    const updateProfileQuery = `
    UPDATE User 
    SET Username = ?, ProfileImageURL = ?, Birthday = ?, Gender = ?, Status = 'active' 
    WHERE UserID = ?`;

    pool.query(
      updateProfileQuery,
      [newUsername, picture, formattedBirthday, gender, userId],
      (err) => {
        if (err) {
          console.error("Error updating profile: ", err);
          return res.status(500).json({ message: "Error updating profile" });
        }

        // ดึงข้อมูลผู้ใช้เพื่อนำไปสร้าง Token
        pool.query(
          "SELECT UserID, Email, Username, ProfileImageURL, Gender FROM User WHERE UserID = ?",
          [userId],
          (err, userResults) => {
            if (err) {
              console.error("Database error fetching user data:", err);
              return res
                .status(500)
                .json({ message: "Error fetching user data" });
            }

            if (userResults.length === 0) {
              return res
                .status(404)
                .json({ message: "User not found after profile update" });
            }

            const user = userResults[0];

            // สร้าง JWT Token
            const token = jwt.sign({ id: user.UserID }, JWT_SECRET, {
              expiresIn: "7d",
            });

            return res.status(200).json({
              message: "Profile set successfully. You are now logged in.",
              token,
              user: {
                id: user.UserID,
                email: user.Email,
                username: user.Username,
                profileImage: user.ProfileImageURL,
                gender: user.Gender,
              },
            });
          }
        );
      }
    );
  }
);

// ---- Search ---- //
// GET /api/search — ใช้ราคาล่าสุดจาก TradingView ถ้ามี
app.get("/api/search", async (req, res) => {
  try {
    const { query } = req.query;
    if (!query) {
      return res.status(400).json({ error: "Search query is required" });
    }

    const searchValue = `%${query.trim().toLowerCase()}%`;

    // ✅ ใช้วันล่าสุดที่ Volume > 0 เป็น baseline (เหมือน favorites)
    const searchSql = `
      SELECT 
          s.StockSymbol, 
          s.Market, 
          s.CompanyName, 
          sd.StockDetailID,         -- แถวล่าสุดที่ Volume>0
          sd.Date, 
          sd.ClosePrice,
          sd.ChangePercen
      FROM Stock s
      INNER JOIN StockDetail sd 
          ON s.StockSymbol = sd.StockSymbol
      INNER JOIN (
          SELECT StockSymbol, MAX(Date) AS LatestDate
          FROM StockDetail
          WHERE Volume > 0
          GROUP BY StockSymbol
      ) latest 
          ON sd.StockSymbol = latest.StockSymbol 
         AND sd.Date = latest.LatestDate
      WHERE LOWER(s.StockSymbol) LIKE ? 
         OR LOWER(s.CompanyName) LIKE ?
      AND sd.Volume > 0
      ORDER BY sd.Date DESC;
    `;

    // 1) ค้นหาหุ้นที่แมตช์
    const [rows] = await pool
      .promise()
      .query(searchSql, [searchValue, searchValue]);
    if (!rows || rows.length === 0) {
      return res.status(404).json({ message: "No results found" });
    }

    // helper: % เปลี่ยนแปลงแบบปลอดภัย (เหมือน favorites)
    const percentChange = (live, base) => {
      const l = Number(live);
      const b = Number(base);
      if (!Number.isFinite(l) || !Number.isFinite(b) || b <= 0) return null;
      return Math.round(((l - b) / b) * 100 * 100) / 100; // ปัด 2 ตำแหน่ง
    };

    // 2) ยิง TradingView ขนานต่อรายการ (ใช้ normalized symbol และ market เดิม)
    const enriched = await Promise.all(
      rows.map(async (stock) => {
        const rawSymbol = stock.StockSymbol || "";
        const normalizedSymbol = rawSymbol.replace(/\.BK$/i, "").toUpperCase();
        const tradingViewMarket =
          stock.Market === "America" ? "usa" : "thailand";

        // baseline = ราคาปิดล่าสุดจาก DB (Volume>0)
        const baselineClose = Number(stock.ClosePrice);
        let livePrice = null;

        try {
          const priceData = await getTradingViewPrice(
            normalizedSymbol,
            tradingViewMarket
          );
          const p = Number(priceData?.price);
          if (Number.isFinite(p) && p > 0) livePrice = p;
        } catch (e) {
          console.error("TradingView API error:", e?.message || e);
        }

        // ✅ ทำเหมือน favorites: คิด % จาก live เทียบ baseline (ถ้าไม่มี live -> null)
        const changePct =
          livePrice != null ? percentChange(livePrice, baselineClose) : null;

        // ✅ ไม่เอา live ไปแทน ClosePrice — คง baseline close ไว้
        return {
          StockSymbol: stock.StockSymbol,
          Market: stock.Market,
          CompanyName: stock.CompanyName,
          StockDetailID: stock.StockDetailID,
          LatestDate: stock.Date,
          ClosePrice: Number.isFinite(baselineClose) ? baselineClose : null,
          ChangePercen: changePct, // % จาก live เทียบ baseline แบบ favorites
          // หากอยากส่งราคาสดประกอบ UI เพิ่มได้ (คอมเมนต์ไว้)
          // LivePrice: livePrice
        };
      })
    );

    res.json({ results: enriched });
  } catch (err) {
    console.error("Error in /api/search:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---- Profile ---- //
//แก้ไขโปรไฟล์
app.put(
  "/api/users/:userId/profile",
  verifyToken,
  upload.single("profileImage"),
  (req, res) => {
    const userId = req.params.userId;

    // ดึงข้อมูลจาก request body
    let { username, gender, birthday } = req.body;
    const profileImage = req.file ? `/uploads/${req.file.filename}` : null;

    // ตรวจสอบว่าค่าที่จำเป็นถูกส่งมาครบหรือไม่
    if (!username || !gender || !birthday) {
      return res
        .status(400)
        .json({ error: "Fields required: username, gender, and birthday" });
    }

    // ตรวจสอบรูปแบบวันเกิดให้ถูกต้อง
    if (isNaN(Date.parse(birthday))) {
      return res
        .status(400)
        .json({ error: "Invalid birthday format (YYYY-MM-DD expected)" });
    }

    // แปลงรูปแบบวันเกิดให้เป็น YYYY-MM-DD
    birthday = formatDateForSQL(birthday);

    // คำนวณอายุจากวันเกิด
    const age = calculateAge(birthday);

    // เช็คว่า Username ถูกใช้โดยผู้ใช้อื่นหรือไม่
    const checkUsernameSql = `SELECT UserID FROM User WHERE Username = ? AND UserID != ?`;

    pool.query(
      checkUsernameSql,
      [username, userId],
      (checkError, checkResults) => {
        if (checkError) {
          console.error("Error checking username:", checkError);
          return res
            .status(500)
            .json({ error: "Database error while checking username" });
        }

        if (checkResults.length > 0) {
          return res.status(400).json({ error: "Username is already in use" });
        }

        // อัปเดตโปรไฟล์ของผู้ใช้ในฐานข้อมูล
        let updateProfileSql = `UPDATE User SET Username = ?, Gender = ?, Birthday = ?`;
        const updateData = [username, gender, birthday];

        // ถ้ามีการอัปโหลดรูปภาพให้เพิ่มเข้าไปใน SQL
        if (profileImage) {
          updateProfileSql += `, ProfileImageURL = ?`;
          updateData.push(profileImage);
        }

        updateProfileSql += ` WHERE UserID = ?;`;
        updateData.push(userId);

        // ทำการอัปเดตข้อมูล
        pool.query(updateProfileSql, updateData, (error, results) => {
          if (error) {
            console.error("Error updating profile:", error);
            return res
              .status(500)
              .json({ error: "Database error while updating user profile" });
          }

          if (results.affectedRows === 0) {
            return res.status(404).json({ error: "User not found" });
          }

          // ส่งข้อมูลที่อัปเดตกลับไปพร้อมอายุที่คำนวณ
          res.json({
            message: "Profile updated successfully",
            userProfile: {
              userId,
              username,
              gender,
              birthday,
              age,
              profileImage: profileImage || "No image uploaded",
            },
          });
        });
      }
    );
  }
);

//ดึงข้อมูลโปรไฟล์
app.get("/api/users/:userId/profile", verifyToken, (req, res) => {
  const userId = req.params.userId;

  const getUserProfileSql = `
    SELECT UserID, Username, Email, Gender, Birthday, ProfileImageURL 
    FROM User WHERE UserID = ?;
  `;

  pool.query(getUserProfileSql, [userId], (error, results) => {
    if (error) {
      console.error("Error retrieving user profile:", error);
      return res
        .status(500)
        .json({ error: "Database error while retrieving user profile" });
    }

    if (results.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const user = results[0];

    // ✅ ตรวจสอบค่า birthday และ profileImage ก่อนส่งออกไป
    const age = user.Birthday ? calculateAge(user.Birthday) : "N/A";
    const profileImage = user.ProfileImageURL
      ? user.ProfileImageURL
      : "/uploads/default.png";

    res.json({
      userId: user.UserID,
      username: user.Username,
      email: user.Email,
      gender: user.Gender,
      birthday: user.Birthday,
      age: age,
      profileImage: profileImage,
    });
  });
});

// Helper function: แปลงวันเกิดเป็น YYYY-MM-DD
function formatDateForSQL(dateString) {
  const dateObj = new Date(dateString);
  const year = dateObj.getFullYear();
  const month = String(dateObj.getMonth() + 1).padStart(2, "0"); // Ensure 2 digits
  const day = String(dateObj.getDate()).padStart(2, "0"); // Ensure 2 digits
  return `${year}-${month}-${day}`;
}

// Helper function: คำนวณอายุจากวันเกิด
function calculateAge(birthday) {
  const birthDate = new Date(birthday);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  if (
    monthDiff < 0 ||
    (monthDiff === 0 && today.getDate() < birthDate.getDate())
  ) {
    age--; // ลดอายุลง 1 ถ้ายังไม่ถึงวันเกิดปีนี้
  }
  return age;
}

app.post("/api/update-fcm-token", verifyToken, async (req, res) => {
  try {
    const userId = req.userId;

    let fcm_token = null; // ค่าดีฟอลต์คือว่าง (NULL ใน DB)
    if (Object.prototype.hasOwnProperty.call(req.body, "fcm_token")) {
      const raw = req.body.fcm_token;
      if (raw && typeof raw === "string" && raw.trim() !== "") {
        fcm_token = raw.trim(); // ถ้ามีค่าจริงก็ใช้ค่านั้น
      }
    }

    // อัปเดต token (จะเป็น NULL ถ้าไม่มีส่งมา)
    await pool_notification.query(
      "UPDATE user SET fcm_token = ? WHERE UserID = ?",
      [fcm_token, userId]
    );

    return res.json({
      message: fcm_token
        ? "อัปเดต fcm_token เรียบร้อย"
        : "ลบ fcm_token (ตั้งเป็นว่าง) เรียบร้อย",
      fcm_token,
    });
  } catch (err) {
    console.error("❌ Error updating fcm_token:", err);
    res.status(500).json({ error: "เกิดข้อผิดพลาดในระบบ" });
  }
});

// แนะนำให้มี UNIQUE KEY ในตาราง notification แบบนี้ (รันครั้งเดียวใน DB)
// ALTER TABLE notification ADD UNIQUE KEY uniq_user_news (UserID, NewsID);

app.get("/api/news-notifications", async (req, res) => {
  try {
    // 1) ดึง "ข่าวล่าสุดที่ยังไม่เคยส่ง" ต่อผู้ใช้แต่ละคน
    const [rows] = await pool_notification.query(`
      WITH ranked AS (
        SELECT
          u.UserID,
          u.fcm_token,
          ns.StockSymbol,
          n.NewsID,
          n.Title,
          n.PublishedDate,
          ROW_NUMBER() OVER (
            PARTITION BY u.UserID
            ORDER BY n.PublishedDate DESC
          ) AS rn
        FROM user u
        JOIN followedstocks f ON f.UserID = u.UserID
        JOIN newsstock ns     ON ns.StockSymbol = f.StockSymbol
        JOIN News n           ON n.NewsID = ns.NewsID
        WHERE u.fcm_token IS NOT NULL
          AND u.fcm_token <> ''
          AND NOT EXISTS (
            SELECT 1
            FROM notification nf
            WHERE nf.UserID = u.UserID
              AND nf.NewsID = n.NewsID
          )
      )
      SELECT
        UserID,
        fcm_token,
        StockSymbol,
        NewsID,
        Title,
        PublishedDate
      FROM ranked
      WHERE rn = 1
    `);

    if (!rows || rows.length === 0) {
      return res.json({
        message:
          "ยังไม่มีข่าวใหม่สำหรับผู้ใช้ (หรือยังไม่มี fcm_token ที่ใช้งานได้)",
        successCount: 0,
        failureCount: 0,
        totalTargets: 0,
        prunedInvalidTokens: 0,
      });
    }

    // 2) กันซ้ำ token ในรอบนี้ (เผื่อผู้ใช้มีหลายหุ้น match พร้อมกัน)
    const seen = new Set();
    const perUserLatest = rows.filter((r) => {
      const t = String(r.fcm_token || "").trim();
      if (!t) return false;
      if (seen.has(t)) return false;
      seen.add(t);
      return true;
    });

    // 3) บันทึก notification แบบ atomic + กันซ้ำระดับ DB ด้วย INSERT IGNORE
    const conn = await pool_notification.getConnection();
    try {
      await conn.beginTransaction();

      for (const r of perUserLatest) {
        // ต้องมี UNIQUE KEY (UserID, NewsID) ที่ตาราง notification เพื่อให้ INSERT IGNORE ทำงานกันซ้ำ
        await conn.query(
          `INSERT IGNORE INTO notification (Message, Date, NewsID, UserID)
           VALUES (?, NOW(), ?, ?)`,
          [r.Title ?? "ข่าวล่าสุด", r.NewsID, r.UserID]
        );
      }

      await conn.commit();
    } catch (e) {
      await conn.rollback();
      throw e;
    } finally {
      conn.release();
    }

    // 4) เตรียม payload สำหรับ FCM
    const makeMessage = (r) => ({
      token: String(r.fcm_token).trim(),
      notification: {
        title: "📰 ข่าวล่าสุด",
        body: r.Title ?? "ข่าวล่าสุด",
      },
      data: {
        userId: String(r.UserID),
        newsId: String(r.NewsID),
        stockSymbol: String(r.StockSymbol ?? ""),
        publishedDate: r.PublishedDate ? String(r.PublishedDate) : "",
      },
      android: { priority: "high" },
      apns: {
        headers: { "apns-priority": "10" },
        payload: { aps: { sound: "default" } },
      },
    });

    const messagesAll = perUserLatest.map(makeMessage);

    // 5) ส่งเป็นชุด ๆ (แนะนำ batch ละ <= 500)
    const chunk = (arr, size) => {
      const out = [];
      for (let i = 0; i < arr.length; i += size)
        out.push(arr.slice(i, i + size));
      return out;
    };
    const batches = chunk(messagesAll, 500);

    const messaging = admin.messaging();
    let successCount = 0;
    let failureCount = 0;
    const invalidTokens = [];

    for (const batch of batches) {
      const results = await Promise.allSettled(
        batch.map((msg) => messaging.send(msg))
      );
      results.forEach((r, i) => {
        if (r.status === "fulfilled") {
          successCount += 1;
        } else {
          failureCount += 1;
          const err = r.reason || {};
          const code = err.code;
          if (
            code === "messaging/registration-token-not-registered" ||
            code === "messaging/invalid-registration-token"
          ) {
            invalidTokens.push(batch[i].token);
          }
        }
      });
    }

    // 6) เคลียร์ token เสียออกจาก user
    if (invalidTokens.length > 0) {
      await pool_notification.query(
        `UPDATE user
         SET fcm_token = NULL
         WHERE fcm_token IN (${invalidTokens.map(() => "?").join(",")})`,
        invalidTokens
      );
    }

    console.log("✅ Notifications sent:", {
      successCount,
      failureCount,
      total: messagesAll.length,
      invalidTokens: invalidTokens.length,
    });

    return res.json({
      message:
        "📤 ส่ง Push notification ข่าวล่าสุดตามหุ้นที่ผู้ใช้ติดตาม สำเร็จ",
      successCount,
      failureCount,
      totalTargets: messagesAll.length,
      prunedInvalidTokens: invalidTokens.length,
    });
  } catch (err) {
    console.error("❌ Error pushing notifications:", err);
    return res
      .status(500)
      .json({ error: "เกิดข้อผิดพลาดขณะดึงข่าวหรือส่ง noti" });
  }
});

app.get("/api/latest-notification", verifyToken, async (req, res) => {
  try {
    const userId = req.userId;

    const [results] = await pool_notification.query(
      `
      SELECT n.NotificationID, n.Message, n.Date, n.NewsID, n.UserID, nw.Title AS NewsTitle, nw.PublishedDate
      FROM notification n
      LEFT JOIN News nw ON n.NewsID = nw.NewsID
      WHERE n.UserID = ?
      ORDER BY n.Date DESC
      LIMIT 10;
    `,
      [userId]
    );

    if (results.length === 0) {
      return res.json({ message: "ไม่มีการแจ้งเตือนล่าสุด" });
    }

    res.json({
      notifications: results,
    });
  } catch (error) {
    console.error("Error fetching latest notifications:", error);
    res.status(500).json({ error: "เกิดข้อผิดพลาดขณะดึงข้อมูลการแจ้งเตือน" });
  }
});

// ดึงข้อมูลข่าวและ push noti
// app.get("/api/news-notifications", verifyToken, async (req, res) => {
//   const today = new Date().toISOString().split("T")[0];

//   const fetchNewsNotificationsSql = `
//     SELECT
//       n.NewsID,
//       n.Title,
//       n.PublishedDate
//     FROM News n
//     WHERE DATE(n.PublishedDate) = ?
//     ORDER BY n.PublishedDate DESC;
//   `;

//   try {
//     const [newsResults] = await pool
//       .promise()
//       .query(fetchNewsNotificationsSql, [today]);

//     // ถ้ามีข่าวใหม่
//     if (newsResults.length > 0) {
//       const latestNews = newsResults[0]; // ข่าวล่าสุด

//       // ดึงรายชื่อผู้ใช้พร้อม FCM Token
//       userResults =
//         "fLmzIwKYS2SuSkidLdjGjs:APA91bFnyXm3-myy4U3Eg1yjwR4ahvtmgHdwLHP4WD-e0StfE4ws6A6oP-cn0HkqW_8YN7mwxpCi4-aScGF_kdjI2chdhQmYxkvkpWCfMSVmt1hCz6Vzf8Q";
//       ฝฝconst[userResults] = await pool
//         .promise()
//         .query("SELECT fcm_token FROM users WHERE fcm_token IS NOT NULL");

//       // สร้าง message สำหรับแต่ละ user
//       const messages = userResults.map((user) => ({
//         notification: {
//           title: "📰 ข่าวสารวันนี้",
//           body: latestNews.Title,
//         },
//         token: user.fcm_token,
//       }));

//       // ส่ง FCM ครั้งละหลาย token ด้วย sendAll
//       const response = await admin.messaging().sendAll(messages);
//       console.log("✅ Notifications sent:", response.successCount, "successes");

//       res.json({
//         message: "📤 Push notification ส่งสำเร็จ",
//         successCount: response.successCount,
//         totalUsers: userResults.length,
//         news: latestNews,
//       });
//     } else {
//       res.json({ message: "ไม่มีข่าวใหม่ในวันนี้", date: today });
//     }
//   } catch (err) {
//     console.error("❌ Error pushing notifications:", err);
//     res.status(500).json({ error: "เกิดข้อผิดพลาดขณะดึงข่าวหรือส่ง noti" });
//   }
// });

// แทนที่ของเดิมทั้ง handler นี้
app.get("/api/news-notifications", async (req, res) => {
  try {
    // เลือก “ข่าวของวันนี้ (เวลาไทย)” ที่แมตช์หุ้นที่ผู้ใช้ติดตาม
    const [rows] = await pool_notification.query(`
      WITH ranked AS (
        SELECT
          u.UserID,
          u.fcm_token,
          ns.StockSymbol,
          n.NewsID,
          n.Title,
          -- ถ้า PublishedDate ใน DB เป็น UTC แปลงมา +07:00 ก่อนค่อยตัดเป็น DATE
          CONVERT_TZ(n.PublishedDate,'+00:00','+07:00') AS PublishedLocal,
          ROW_NUMBER() OVER (
            PARTITION BY u.UserID
            ORDER BY n.PublishedDate DESC
          ) rn
        FROM user u
        JOIN followedstocks f ON f.UserID = u.UserID
        JOIN newsstock ns ON ns.StockSymbol = f.StockSymbol
        JOIN News n ON n.NewsID = ns.NewsID
        WHERE u.fcm_token IS NOT NULL AND u.fcm_token <> ''
          AND DATE(CONVERT_TZ(n.PublishedDate,'+00:00','+07:00')) = CURDATE()
      )
      SELECT UserID, fcm_token, StockSymbol, NewsID, Title, PublishedLocal
      FROM ranked
      WHERE rn = 1
    `);

    if (!rows || rows.length === 0) {
      return res.json({ message: "วันนี้ยังไม่พบข่าวที่ตรงกับหุ้นที่ติดตาม" });
    }

    // กัน token ซ้ำ (ถ้าผู้ใช้บังเอิญมี token ซ้ำกันหลายแถว)
    const seen = new Set();
    const targets = rows.filter((r) => {
      const t = String(r.fcm_token || "").trim();
      if (!t || seen.has(t)) return false;
      seen.add(t);
      return true;
    });

    // บันทึก log แจ้งเตือน (option)
    const conn = await pool_notification.getConnection();
    try {
      await conn.beginTransaction();
      for (const r of targets) {
        await conn.query(
          `INSERT INTO notification (Message, Date, NewsID, UserID)
           VALUES (?, NOW(), ?, ?)`,
          [r.Title ?? "ข่าวล่าสุด", r.NewsID, r.UserID]
        );
      }
      await conn.commit();
    } catch (e) {
      await conn.rollback();
      throw e;
    } finally {
      conn.release();
    }

    // ส่ง FCM (batch ทีละ 500 หรือจะส่งทีละ 1 ก็ได้)
    const messaging = admin.messaging();
    const messages = targets.map((r) => ({
      token: r.fcm_token.trim(),
      notification: { title: "📰 ข่าววันนี้", body: r.Title ?? "ข่าวล่าสุด" },
      data: {
        userId: String(r.UserID),
        newsId: String(r.NewsID),
        symbol: String(r.StockSymbol ?? ""),
        publishedDate: r.PublishedLocal
          ? new Date(r.PublishedLocal).toISOString()
          : "",
      },
      android: {
        priority: "high",
        notification: { channelId: "news", sound: "default" },
      },
      apns: { payload: { aps: { sound: "default" } } },
    }));

    // ส่งแบบขนานและนับผลลัพธ์
    const chunk = (arr, size) =>
      arr.reduce(
        (acc, _, i) => (i % size ? acc : [...acc, arr.slice(i, i + size)]),
        []
      );
    const batches = chunk(messages, 500);

    let successCount = 0;
    let failureCount = 0;
    const invalidTokens = [];

    for (const batch of batches) {
      const results = await Promise.allSettled(
        batch.map((m) => messaging.send(m))
      );
      results.forEach((r, i) => {
        if (r.status === "fulfilled") successCount++;
        else {
          failureCount++;
          const code = r.reason?.code;
          if (
            code === "messaging/registration-token-not-registered" ||
            code === "messaging/invalid-registration-token"
          ) {
            invalidTokens.push(batch[i].token);
          }
        }
      });
    }

    if (invalidTokens.length) {
      await pool_notification.query(
        `UPDATE user SET fcm_token = NULL WHERE fcm_token IN (${invalidTokens
          .map(() => "?")
          .join(",")})`,
        invalidTokens
      );
    }

    return res.json({
      message: "📤 ส่งข่าววันนี้สำเร็จ",
      successCount,
      failureCount,
      totalTargets: messages.length,
      prunedInvalidTokens: invalidTokens.length,
    });
  } catch (err) {
    console.error("❌ /api/news-notifications error:", err);
    return res.status(500).json({ error: "เกิดข้อผิดพลาดขณะส่งข่าววันนี้" });
  }
});

// ---- Favorites ---- //

// API สำหรับเพิ่มบุ๊คมาร์ค
app.post("/api/favorites", verifyToken, (req, res) => {
  const { stock_symbol } = req.body; // ดึง StockSymbol จาก request body
  const user_id = req.userId; // ดึง user_id จาก Token ที่ผ่านการตรวจสอบแล้ว

  // ตรวจสอบว่ามี StockSymbol หรือไม่
  if (!stock_symbol) {
    return res.status(400).json({ error: "Stock symbol is required" });
  }

  // ตรวจสอบว่าผู้ใช้ติดตามหุ้นนี้ไปแล้วหรือยัง
  const checkFollowSql =
    "SELECT * FROM FollowedStocks WHERE UserID = ? AND StockSymbol = ?";
  pool.query(checkFollowSql, [user_id, stock_symbol], (err, results) => {
    if (err) {
      console.error("Database error during checking followed stock:", err);
      return res
        .status(500)
        .json({ error: "Database error during checking followed stock" });
    }

    if (results.length > 0) {
      return res
        .status(400)
        .json({ error: "You are already following this stock" });
    }

    // เพิ่มข้อมูลหุ้นที่ติดตามลงในตาราง FollowedStocks
    const followStockSql =
      "INSERT INTO FollowedStocks (UserID, StockSymbol, FollowDate) VALUES (?, ?, NOW())";
    pool.query(followStockSql, [user_id, stock_symbol], (err) => {
      if (err) {
        console.error("Database error during following stock:", err);
        return res.status(500).json({ error: "Error following stock" });
      }

      res.status(201).json({ message: "Stock followed successfully" });
    });
  });
});

// API สำหรับเลิกติดตามหุ้น
app.delete("/api/favorites", verifyToken, (req, res) => {
  const { stock_symbol } = req.body; // ดึง StockSymbol จาก request body
  const user_id = req.userId; // ดึง user_id จาก Token ที่ผ่านการตรวจสอบแล้ว

  // ตรวจสอบว่ามี StockSymbol ที่ต้องการลบหรือไม่
  if (!stock_symbol) {
    return res.status(400).json({ error: "Stock symbol is required" });
  }

  // ลบข้อมูลหุ้นที่ติดตามจากฐานข้อมูล
  const deleteFollowedStockSql =
    "DELETE FROM FollowedStocks WHERE UserID = ? AND StockSymbol = ?";
  pool.query(
    deleteFollowedStockSql,
    [user_id, stock_symbol],
    (err, results) => {
      if (err) {
        console.error("Database error during unfollowing stock:", err);
        return res.status(500).json({ error: "Error unfollowing stock" });
      }

      if (results.affectedRows === 0) {
        return res.status(404).json({
          message:
            "Stock not found in followed list or you are not authorized to remove",
        });
      }

      res.json({ message: "Stock unfollowed successfully" });
    }
  );
});

app.get("/api/favorites", verifyToken, async (req, res) => {
  const userId = req.userId;

  const normalize = (sym) =>
    String(sym || "")
      .replace(/\.BK$/i, "")
      .toUpperCase();

  // 1) หุ้นที่ผู้ใช้ follow + ข้อมูลตลาด
  const fetchFavoritesSql = `
    SELECT fs.FollowID, fs.StockSymbol, fs.FollowDate, s.CompanyName, s.Market
    FROM FollowedStocks fs
    JOIN Stock s ON fs.StockSymbol = s.StockSymbol
    WHERE fs.UserID = ?
    ORDER BY fs.FollowDate DESC
  `;

  // 2) baseline: วันล่าสุดที่ Volume > 0
  const fetchLatestCloseSql = `
    SELECT sd.StockSymbol, sd.ClosePrice AS LatestClose, sd.Date AS LatestCloseDate
    FROM StockDetail sd
    JOIN (
      SELECT StockSymbol, MAX(Date) AS LatestDate
      FROM StockDetail
      WHERE StockSymbol IN (?) AND Volume > 0
      GROUP BY StockSymbol
    ) t
      ON sd.StockSymbol = t.StockSymbol
     AND sd.Date = t.LatestDate
    WHERE sd.Volume > 0
  `;

  // helper: % เปลี่ยนแปลงแบบปลอดภัย
  const percentChange = (live, base) => {
    const l = Number(live);
    const b = Number(base);
    if (!Number.isFinite(l) || !Number.isFinite(b) || b <= 0) return null;
    return Math.round(((l - b) / b) * 100 * 100) / 100;
  };

  try {
    // 1) รายการที่ follow
    const [stockResults] = await pool
      .promise()
      .query(fetchFavoritesSql, [userId]);
    if (!stockResults || stockResults.length === 0) {
      return res.status(404).json({ message: "No followed stocks found" });
    }

    // เตรียมสัญลักษณ์ให้ครอบคลุม
    const originalSyms = stockResults.map((s) => s.StockSymbol);
    const normalizedSyms = originalSyms.map(normalize);
    const searchSyms = Array.from(
      new Set([...originalSyms, ...normalizedSyms])
    );

    // 2) ดึง baseline “วันล่าสุดที่ Volume>0”
    const [latestCloseRows] = await pool
      .promise()
      .query(fetchLatestCloseSql, [searchSyms]);

    // 3) map baseline ด้วยคีย์ normalize
    const latestCloseMap = new Map();
    latestCloseRows.forEach((r) => {
      const key = normalize(r.StockSymbol);
      latestCloseMap.set(key, {
        close: Number(r.LatestClose),
        date: r.LatestCloseDate,
        rawSymbol: r.StockSymbol,
      });
    });

    // 4) เตรียม bucket ผลลัพธ์
    const buckets = {};
    stockResults.forEach((s) => {
      const norm = normalize(s.StockSymbol);
      const lc = latestCloseMap.get(norm);
      buckets[norm] = {
        FollowID: s.FollowID,
        StockSymbol: s.StockSymbol,
        CompanyName: s.CompanyName,
        FollowDate: s.FollowDate,
        Market: s.Market,

        BaselineClose: lc && Number.isFinite(lc.close) ? lc.close : null,
        BaselineDate: lc?.date || null,

        LastPrice: null,
        LastChange: null,

        _MatchedDBSymbol: lc?.rawSymbol || null,

        HistoricalPrices: [],
      };
    });

    // 5) ราคาสดจาก TradingView
    const livePrices = await Promise.all(
      stockResults.map(async (row) => {
        const norm = normalize(row.StockSymbol);
        try {
          const market = row?.Market === "America" ? "usa" : "thailand";
          const priceData = await getTradingViewPrice(norm, market);
          const live = Number(priceData?.price);
          return { norm, live: Number.isFinite(live) ? live : null };
        } catch (e) {
          console.error("TradingView API error:", e?.message || e);
          return { norm, live: null };
        }
      })
    );

    // 6) คำนวณ %
    livePrices.forEach(({ norm, live }) => {
      const b = buckets[norm];
      if (!b) return;
      if (live != null) b.LastPrice = live;
      b.LastChange = percentChange(live, b.BaselineClose);
    });

    // 7) ส่งกลับตามลำดับเดิม
    const result = stockResults.map((s) => buckets[normalize(s.StockSymbol)]);
    return res.json(result);
  } catch (err) {
    console.error("Error in /api/favorites:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// สมมติว่ามีฟังก์ชันนี้อยู่แล้ว
// async function getTradingViewPrice(symbol, market) -> { price: number }
// ✅ /api/top-10-stocks
// เงื่อนไขตามที่ต้องการ:
// - PredictionClose_Ensemble = ราคาพรุ่งนี้
// - ถ้า Prediction < ราคาปัจจุบัน (TradingView close) => ExpectedDiffPercent ติดลบ (หุ้นคาดว่าจะ "ลง")
// - ส่งตัวเลขเป็น string ทั้งหมด (เช่น "123.45")

// ✅ /api/top-10-stocks
// - ChangePercentage คิดใหม่จาก PredictionClose_Ensemble vs ClosePrice
//   ((Prediction - Close) / Close) * 100  → เป็นลบเมื่อ Prediction < Close
// - ตัวเลขส่งออกเป็น string

// ✅ /api/top-10-stocks
// - เลือกวันล่าสุด
// - คำนวณ ChangePercentageSigned = ((Prediction - Close) / Close) * 100 ใน SQL
// - จัดอันดับ DESC เอา 10 ตัวแรก (ขึ้นมากสุด = ค่าบวกมากสุด)
// - ดึงราคาปัจจุบัน (TradingView) มาเสริม และคำนวณ expected พรุ่งนี้จาก Prediction - Current
// - ตัวเลขทั้งหมดส่งเป็น string

app.get("/api/top-10-stocks", async (req, res) => {
  try {
    const conn = await pool.promise().getConnection();
    try {
      // 1) วันล่าสุด
      const [dateRows] = await conn.query(
        "SELECT MAX(Date) AS LatestDate FROM StockDetail"
      );
      const latestDate = dateRows?.[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // 2) ดึงทั้งวันล่าสุด (ไปคัด+เรียงใน JS จาก 'ราคาสด')
      const [rows] = await conn.query(
        `
        SELECT 
          sd.StockDetailID,
          s.StockSymbol,
          s.Market,
          sd.PredictionClose_Ensemble,
          sd.ClosePrice,
          (
            (sd.PredictionClose_Ensemble - sd.ClosePrice) / sd.ClosePrice
          ) * 100 AS ChangePercentageSigned
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ?
        `,
        [latestDate]
      );

      // helpers
      const round = (n, d = 2) =>
        Number.isFinite(n) ? Number(n.toFixed(d)) : null;
      const str = (n, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : null);

      // 3) enrich จาก TradingView + คัดเฉพาะที่ Ensemble > ราคาสด (แนวโน้มขึ้น)
      const enrichedWithKey = (
        await Promise.all(
          rows.map(async (r) => {
            const rawSymbol = r.StockSymbol || "";
            const normalizedSymbol = rawSymbol
              .replace(/\.BK$/i, "")
              .toUpperCase();
            const market = r.Market === "America" ? "usa" : "thailand";

            const prediction = Number(r.PredictionClose_Ensemble);
            const closePrice = Number(r.ClosePrice);

            // ราคาปัจจุบันจาก TradingView (fallback -> close)
            let currentPrice = null;
            try {
              const live = await getTradingViewPrice(normalizedSymbol, market);
              const p = Number(live?.price);
              if (Number.isFinite(p) && p > 0) currentPrice = p;
            } catch (e) {
              console.error(
                `TradingView error for ${normalizedSymbol}:`,
                e?.message || e
              );
            }
            if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
              currentPrice =
                Number.isFinite(closePrice) && closePrice > 0
                  ? closePrice
                  : null;
            }

            // เปอร์เซ็นต์จาก Prediction vs Close (SIGNED) — ค่าจาก SQL (สำรองไว้ใช้ fallback)
            const changePctSignedSQL = Number(r.ChangePercentageSigned);

            // Expected จาก Prediction - Current (SIGNED, ฐาน Current)
            let expectedDiffAbs = null;
            let expectedDiffPctSigned = null;
            let expectedDirection = null;
            let UpPotentialPercent = null;
            let DownRiskPercent = null;
            let sortKey = -Infinity; // ใช้เรียงภายในเท่านั้น

            if (
              Number.isFinite(prediction) &&
              Number.isFinite(currentPrice) &&
              currentPrice > 0
            ) {
              const delta = prediction - currentPrice; // >0 คาดขึ้น, <0 คาดลง
              expectedDiffAbs = round(delta, 4);
              expectedDiffPctSigned = round((delta / currentPrice) * 100, 2);

              if (delta > 0) {
                expectedDirection = "UP";
                UpPotentialPercent = expectedDiffPctSigned;
                DownRiskPercent = 0;
                sortKey = expectedDiffPctSigned; // เรียงมาก -> น้อย
              } else if (delta < 0) {
                expectedDirection = "DOWN";
                UpPotentialPercent = 0;
                DownRiskPercent = round(
                  Math.abs((delta / currentPrice) * 100),
                  2
                );
                sortKey = -Infinity; // ไม่คัดลง
              } else {
                expectedDirection = "FLAT";
                UpPotentialPercent = 0;
                DownRiskPercent = 0;
                sortKey = -Infinity;
              }
            }

            // ✅ เปลี่ยน 'ChangePercentage' ให้เป็น Prediction vs Current (ฐาน Current)
            // ถ้าคำนวณไม่ได้ (เช่นไม่มีราคาสดและ close เป็น null) จะ fallback ไปใช้ค่า SQL เดิม
            const changePercentageForOutput = Number.isFinite(
              expectedDiffPctSigned
            )
              ? expectedDiffPctSigned
              : changePctSignedSQL;

            // schema เดิมสำหรับ mobile (ชื่อคีย์เท่าเดิม)
            const data = {
              StockDetailID: r.StockDetailID,
              StockSymbol: r.StockSymbol,
              Market: r.Market,

              ClosePrice: str(closePrice, 2),
              PredictionClose_Ensemble: str(
                Number(r.PredictionClose_Ensemble),
                2
              ),

              // ✅ ตอนนี้เป็น Prediction vs Current แล้ว
              ChangePercentage: str(changePercentageForOutput, 2),

              // ราคาปัจจุบัน (แสดงประกอบ)
              CurrentPrice: str(currentPrice, 2),

              // สัญญาณพรุ่งนี้ (Prediction เทียบ Current)
              ExpectedDiffPrice: str(expectedDiffAbs, 4),
              ExpectedDiffPercent: str(expectedDiffPctSigned, 2),
              ExpectedDirection: expectedDirection,

              UpPotentialPercent: str(UpPotentialPercent, 2),
              DownRiskPercent: str(DownRiskPercent, 2),
            };

            return { data, sortKey };
          })
        )
      )
        .filter((x) => Number.isFinite(x.sortKey) && x.sortKey > 0) // แนวโน้มขึ้น
        .sort((a, b) => b.sortKey - a.sortKey) // มาก -> น้อย
        .slice(0, 10)
        .map((x) => x.data);

      return res.json({ date: latestDate, topStocks: enrichedWithKey });
    } finally {
      conn.release();
    }
  } catch (error) {
    console.error("Internal server error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// API สำหรับดึง 3 หุ้นที่มีการเปลี่ยนแปลงสูงสุด พร้อมข้อมูลย้อนหลัง 5 วัน
app.get("/api/trending-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุดที่มีข้อมูล
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res
          .status(500)
          .json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0].LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่หุ้นที่มีการเปลี่ยนแปลงสูงสุด 3 อันดับแรก
      const trendingStocksQuery = `
        SELECT 
          sd.StockDetailID,
          sd.Date, 
          s.StockSymbol, 
          sd.Changepercen AS ChangePercentage, 
          sd.ClosePrice,
          sd.PredictionClose,
          s.Market
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ?
        ORDER BY s.Market DESC, sd.Changepercen DESC
        LIMIT 3;
      `;

      pool.query(trendingStocksQuery, [latestDate], (err, trendingStocks) => {
        if (err) {
          console.error("Database error fetching trending stocks:", err);
          return res
            .status(500)
            .json({ error: "Database error fetching trending stocks" });
        }

        const stockSymbols = trendingStocks.map((stock) => stock.StockSymbol);

        if (stockSymbols.length === 0) {
          return res.status(404).json({ error: "No trending stocks found" });
        }

        // ดึงข้อมูลย้อนหลัง 5 วัน (นับจากวันล่าสุด)
        const historyQuery = `
          SELECT 
            *
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY Date DESC
          LIMIT ?;
        `;

        pool.query(
          historyQuery,
          [stockSymbols, stockSymbols.length * 5],
          (err, historyData) => {
            if (err) {
              console.error("Database error fetching historical data:", err);
              return res
                .status(500)
                .json({ error: "Database error fetching historical data" });
            }

            // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
            const historyMap = {};
            historyData.forEach((entry) => {
              if (!historyMap[entry.StockSymbol]) {
                historyMap[entry.StockSymbol] = [];
              }
              if (historyMap[entry.StockSymbol].length < 5) {
                historyMap[entry.StockSymbol].push({
                  Date: entry.Date,
                  ClosePrice: entry.ClosePrice,
                });
              }
            });

            // แปลงข้อมูลให้ตรงกับโครงสร้าง JSON ที่ต้องการ
            const response = {
              date: latestDate,
              trendingStocks: trendingStocks.map((stock) => {
                const priceChangePercentage = stock.PredictionClose
                  ? ((stock.PredictionClose - stock.ClosePrice) /
                      stock.ClosePrice) *
                    100
                  : null;

                let stockType =
                  stock.Market === "America" ? "US Stock" : "TH Stock";

                return {
                  StockDetailID: stock.StockDetailID,
                  Date: stock.Date,
                  StockSymbol: stock.StockSymbol,
                  ChangePercentage: stock.ChangePercentage,
                  ClosePrice: stock.ClosePrice,
                  PredictionClose: stock.PredictionClose,
                  PricePredictionChange: priceChangePercentage
                    ? priceChangePercentage.toFixed(2) + "%"
                    : "N/A",
                  Type: stockType,
                  HistoricalPrices: historyMap[stock.StockSymbol] || [],
                };
              }),
            };

            res.json(response);
          }
        );
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});
// ---- News ---- //
// Assume 'app' is your Express app instance and 'pool' is your database connection pool.
// Example:
// const express = require('express');
// const mysql = require('mysql2');
// const app = express();
// const pool = mysql.createPool({ ...your_db_config });

// =============================
// /api/latest-news  (dedupe StockSymbols per news)
// =============================

app.get("/api/latest-news", async (req, res) => {
  try {
    // 1) sanitize params
    const limit = Math.max(1, Math.min(parseInt(req.query.limit) || 20, 100));
    const offset = Math.max(0, parseInt(req.query.offset) || 0);
    const sentimentInput = (req.query.sentiment || "").toString();
    const sortParam = (req.query.sort || "").toString().toUpperCase();
    const sortOrder = sortParam === "ASC" ? "ASC" : "DESC";

    // Accept either `source` or `region` as a friendly filter
    const sourceInput = (req.query.source || req.query.region || "").toString();

    // 2) map friendly -> DB values (normalize once!)
    //    === Make sure these match values stored in News.Source exactly ===
    const sourceMap = new Map([
      ["th", "bangkokpost"],
      ["th news", "bangkokpost"],
      ["bangkokpost", "bangkokpost"],

      ["us", "investing"],
      ["us news", "investing"],
      ["investing", "investing"],
    ]);
    const key = sourceInput.trim().toLowerCase();
    const sourceDbValue = sourceMap.get(key) || null;

    // 3) build WHERE conditions (for subquery)
    const conditions = [];
    const params = [];

    if (sourceDbValue) {
      conditions.push("Source = ?");
      params.push(sourceDbValue);
    }

    if (["Positive", "Negative", "Neutral"].includes(sentimentInput)) {
      conditions.push("Sentiment = ?");
      params.push(sentimentInput);
    }

    const whereClause =
      conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    // 4) subquery (IDs only) for fast pagination
    const subQuery = `
      SELECT NewsID
      FROM News
      ${whereClause}
      ORDER BY PublishedDate ${sortOrder}, NewsID ${sortOrder}
      LIMIT ? OFFSET ?
    `;
    const subQueryParams = [...params, limit, offset];

    // 5) main query joins only selected IDs
    const mainQuery = `
      SELECT
        s.NewsID,
        s.Title,
        s.Source,
        s.Sentiment,
        DATE_FORMAT(s.PublishedDate, '%Y-%m-%d') AS PublishedDate,
        s.Img,
        st.StockSymbol
      FROM News s
      JOIN (${subQuery}) p ON s.NewsID = p.NewsID
      LEFT JOIN Newsstock st ON s.NewsID = st.NewsID
      ORDER BY s.PublishedDate ${sortOrder}, s.NewsID ${sortOrder}
    `;

    pool.query(mainQuery, subQueryParams, (err, rows) => {
      if (err) {
        console.error("DB Error (/api/latest-news):", err);
        return res.status(500).json({ error: "Database error" });
      }

      // 6) group by NewsID and DEDUPE symbols with Set
      const grouped = {};
      rows.forEach((r) => {
        const id = r.NewsID;
        if (!grouped[id]) {
          grouped[id] = {
            NewsID: r.NewsID,
            Title: r.Title,
            Source: r.Source,
            Sentiment: r.Sentiment,
            PublishedDate: r.PublishedDate,
            Img: r.Img,
            StockSymbols: new Set(),
          };
        }
        if (r.StockSymbol) grouped[id].StockSymbols.add(r.StockSymbol);
      });

      // 7) finalize (convert Set -> Array)
      const news = Object.values(grouped).map((n) => ({
        ...n,
        StockSymbols: Array.from(n.StockSymbols), // or .sort()
      }));

      res.json({ news });
    });
  } catch (error) {
    console.error("Server Error (/api/latest-news):", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// =============================
// /api/news-by-source  (JOIN + dedupe StockSymbols)
// =============================

app.get("/api/news-by-source", async (req, res) => {
  try {
    const limit = Math.max(1, Math.min(parseInt(req.query.limit) || 20, 100));
    const offset = Math.max(0, parseInt(req.query.offset) || 0);

    // Accept `region` or `source` from client, normalize to DB value
    const input = (req.query.region || req.query.source || "").toString();

    const sourceMap = new Map([
      ["th", "bangkokpost"],
      ["th news", "bangkokpost"],
      ["bangkokpost", "bangkokpost"],

      ["us", "investing"],
      ["us news", "investing"],
      ["investing", "investing"],
    ]);
    const key = input.trim().toLowerCase();
    const sourceName = sourceMap.get(key);

    if (!sourceName) {
      return res.status(400).json({ error: "Invalid region/source" });
    }

    // JOIN Newsstock เพื่อได้สัญลักษณ์ แล้ว dedupe ต่อข่าว
    const query = `
      SELECT
        n.NewsID,
        n.Title,
        n.Source,
        n.Sentiment,
        DATE_FORMAT(n.PublishedDate, '%Y-%m-%d') AS PublishedDate,
        n.Img,
        st.StockSymbol
      FROM News n
      LEFT JOIN Newsstock st ON n.NewsID = st.NewsID
      WHERE n.Source = ?
      ORDER BY n.PublishedDate DESC, n.NewsID DESC
      LIMIT ? OFFSET ?
    `;

    pool.query(query, [sourceName, limit, offset], (err, rows) => {
      if (err) {
        console.error("DB Error (/api/news-by-source):", err);
        return res.status(500).json({ error: "Database error" });
      }

      // group + dedupe
      const grouped = {};
      rows.forEach((r) => {
        const id = r.NewsID;
        if (!grouped[id]) {
          grouped[id] = {
            NewsID: r.NewsID,
            Title: r.Title,
            Source: r.Source,
            Sentiment: r.Sentiment,
            PublishedDate: r.PublishedDate,
            Img: r.Img,
            StockSymbols: new Set(),
          };
        }
        if (r.StockSymbol) grouped[id].StockSymbols.add(r.StockSymbol);
      });

      const news = Object.values(grouped).map((n) => ({
        ...n,
        StockSymbols: Array.from(n.StockSymbols), // or .sort()
      }));

      res.json({ news });
    });
  } catch (error) {
    console.error("Error (/api/news-by-source):", error);
    res.status(500).json({ error: "Internal error" });
  }
});

app.get("/api/recommentnews-stockdetail", async (req, res) => {
  try {
    const stockSymbol = req.query.symbol;

    if (!stockSymbol) {
      return res
        .status(400)
        .json({ error: "Missing StockSymbol in query parameters" });
    }

    const newsRecommentQuery = `
      SELECT 
        ns.StockSymbol,
        ns.NewsID,
        n.Title,
        n.Source,
        n.PublishedDate,
        n.Sentiment,
        n.Img
      FROM newsstock ns
      JOIN news n ON ns.NewsID = n.NewsID
      WHERE ns.StockSymbol = ?
      ORDER BY n.PublishedDate DESC
      LIMIT 10
    `;

    pool.query(newsRecommentQuery, [stockSymbol], (err, results) => {
      if (err) {
        console.error("Database error fetching news detail:", err);
        return res
          .status(500)
          .json({ error: "Database error fetching news detail" });
      }

      if (results.length === 0) {
        if (results.length === 0) {
          return res.json([]);
        }
        return res.status(404).json({ error: "News not found" });
      }

      res.json(results);
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/recommentnews-stockdetail", async (req, res) => {
  try {
    const stockSymbol = req.query.symbol;

    if (!stockSymbol) {
      return res
        .status(400)
        .json({ error: "Missing StockSymbol in query parameters" });
    }

    const newsRecommentQuery = `
      SELECT 
        ns.StockSymbol,
        ns.NewsID,
        n.Title,
        n.Source,
        n.PublishedDate,
        n.Sentiment,
        n.Img
      FROM newsstock ns
      JOIN news n ON ns.NewsID = n.NewsID
      WHERE ns.StockSymbol = ?
      ORDER BY n.PublishedDate DESC
      LIMIT 10
    `;

    pool.query(newsRecommentQuery, [stockSymbol], (err, results) => {
      if (err) {
        console.error("Database error fetching news detail:", err);
        return res
          .status(500)
          .json({ error: "Database error fetching news detail" });
      }

      if (results.length === 0) {
        if (results.length === 0) {
          return res.json([]);
        }
        return res.status(404).json({ error: "News not found" });
      }

      res.json(results);
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Detail News
app.get("/api/news-detail", async (req, res) => {
  try {
    const { id } = req.query; // ใช้ NewsID แทน Title
    if (!id) {
      return res.status(400).json({ error: "News ID is required" });
    }

    // คิวรี่ดึงรายละเอียดข่าวโดยใช้ NewsID
    const newsDetailQuery = `
      SELECT NewsID, Title, Sentiment, Source, PublishedDate, ConfidenceScore, Content, URL,Img
      FROM News
      WHERE NewsID = ?
      LIMIT 1;
    `;

    pool.query(newsDetailQuery, [id], (err, results) => {
      if (err) {
        console.error("Database error fetching news detail:", err);
        return res
          .status(500)
          .json({ error: "Database error fetching news detail" });
      }

      if (results.length === 0) {
        return res.status(404).json({ error: "News not found" });
      }

      const news = results[0];

      // แปลง ConfidenceScore เป็นเปอร์เซ็นต์
      const confidencePercentage = `${(news.ConfidenceScore * 100).toFixed(
        0
      )}%`;

      res.json({
        NewsID: news.NewsID,
        Title: news.Title,
        Sentiment: news.Sentiment,
        Source: news.Source,
        PublishedDate: news.PublishedDate,
        ConfidenceScore: confidencePercentage,
        Content: news.Content,
        ImageURL: news.Img,
        URL: news.URL,
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ฟังก์ชันดึงค่าเงิน USD → THB จาก API ภายนอก
async function getExchangeRate() {
  try {
    const response = await fetch(
      "https://api.exchangerate-api.com/v4/latest/USD"
    );
    const data = await response.json();
    return data.rates.THB || 1; // ถ้าค่า THB ไม่มีให้คืนค่า 1 (ไม่แปลง)
  } catch (error) {
    console.error("Error fetching exchange rate:", error);
    return 1; // ถ้าดึงค่าไม่ได้ให้ใช้ค่า 1 (ป้องกัน error)
  }
}

// API ดึงรายละเอียดหุ้น
// ฟังก์ชันช่วย format วันที่เป็น YYYY-MM-DD
function formatDateYYYYMMDD(date) {
  const d = new Date(date);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

app.get("/api/stock-detail/:symbol", async (req, res) => {
  const conn = pool.promise();

  // === helpers ===
  const toNum = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  function percentChange(live, base) {
    const l = toNum(live);
    const b = toNum(base);
    if (l == null || b == null || b <= 0) return null;
    return Math.round(((l - b) / b) * 100 * 100) / 100;
  }

  function formatDateYYYYMMDD(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  try {
    const rawSymbol = (req.params.symbol || "").toUpperCase();
    const { timeframe = "5D" } = req.query;

    const historyLimits = {
      "1D": 1,
      "5D": 5,
      "1M": 22,
      "3M": 66,
      "6M": 132,
      "1Y": 264,
      ALL: null,
    };
    if (!Object.prototype.hasOwnProperty.call(historyLimits, timeframe)) {
      return res.status(400).json({
        error: "Invalid timeframe. Choose from 1D, 5D, 1M, 3M, 6M, 1Y, ALL.",
      });
    }

    const symbol = rawSymbol.replace(".BK", "");
    const normalizedSymbol = symbol.toUpperCase();

    // 1) แถวล่าสุด + ข้อมูลบริษัท (✅ กรองวันล่าสุดที่ Volume > 0)
    const latestRowSql = `
      SELECT 
        sd.*,
        s.CompanyName,
        s.Market,
        s.Sector,
        s.Industry,
        s.Description
      FROM StockDetail sd
      JOIN Stock s ON s.StockSymbol = sd.StockSymbol
      WHERE sd.StockSymbol = ?
        AND sd.Date = (
          SELECT MAX(Date)
          FROM StockDetail
          WHERE StockSymbol = ?
            AND Volume > 0
        )
        AND sd.Volume > 0
      LIMIT 1
    `;
    const [latestRows] = await conn.query(latestRowSql, [
      normalizedSymbol,
      normalizedSymbol,
    ]);
    if (!latestRows || latestRows.length === 0) {
      return res.status(404).json({ error: "Stock not found" });
    }
    const stock = latestRows[0];

    // 2) ประเภท + อัตราแลกเปลี่ยน
    const stockType = stock.Market === "America" ? "US Stock" : "TH Stock";
    let usdThb = 1;
    try {
      usdThb = await getExchangeRate();
      if (!Number.isFinite(Number(usdThb)) || Number(usdThb) <= 0) usdThb = 1;
    } catch {
      usdThb = 1;
    }

    // 3) ราคาสดจาก TradingView (เอาเฉพาะ price; baseline มาจาก DB แถวล่าสุดที่ Volume>0)
    let tv = null;
    let livePrice = null;
    try {
      const tvMarket = stock.Market === "America" ? "usa" : "thailand";
      tv = await getTradingViewPrice(normalizedSymbol, tvMarket);
      livePrice = toNum(tv?.price);
    } catch (e) {
      console.error("TradingView API error:", e?.message || e);
      tv = null;
    }

    // === Baseline/เปอร์เซ็นต์: ใช้ DB ล่าสุดที่ Volume>0 (จาก latestRowSql) ===
    const dbClose = toNum(stock.ClosePrice);
    const baselineClose = dbClose ?? null;
    const baselineDate = stock.Date || null;
    const baselineSource = "DB_LAST_CLOSE_VOL>0";

    // ราคา "สำหรับแสดงผล" (ถ้าไม่มี live ก็ใช้ baseline)
    const closePriceNative = livePrice ?? baselineClose ?? 0;

    // แปลงสกุลเพื่อแสดงผล
    const ClosePriceUSD =
      stockType === "US Stock"
        ? closePriceNative
        : usdThb !== 0
        ? closePriceNative / usdThb
        : closePriceNative;

    const ClosePriceTHB =
      stockType === "TH Stock" ? closePriceNative : closePriceNative * usdThb;

    // % เทียบ live vs baseline
    const liveVsBaselinePct =
      livePrice == null ? null : percentChange(livePrice, baselineClose);

    // ส่วนต่าง/ทิศทาง
    let priceDelta = null;
    if (livePrice != null && baselineClose != null) {
      priceDelta = Math.round((livePrice - baselineClose) * 100) / 100;
    }
    let direction = null; // 'UP' | 'DOWN' | 'FLAT' | null
    if (priceDelta != null) {
      direction = priceDelta > 0 ? "UP" : priceDelta < 0 ? "DOWN" : "FLAT";
    }
    const liveVsBaselinePctAbs =
      liveVsBaselinePct == null ? null : Math.abs(liveVsBaselinePct);

    // 4) Prediction (ใช้ closePriceNative เป็นฐานเทียบ)
    const predictionClose = Object.prototype.hasOwnProperty.call(
      stock,
      "PredictionClose"
    )
      ? stock.PredictionClose != null
        ? Number(stock.PredictionClose)
        : null
      : null;

    const predictionTrend = Object.prototype.hasOwnProperty.call(
      stock,
      "PredictionTrend"
    )
      ? stock.PredictionTrend
      : null;

    const pricePredictionChange =
      predictionClose != null && Number(closePriceNative) !== 0
        ? (
            ((predictionClose - closePriceNative) / closePriceNative) *
            100
          ).toFixed(2) + "%"
        : "N/A";

    // 5) Avg Vol 30D
    const avgVolSql = `
      SELECT AVG(Volume) AS AvgVolume30D 
      FROM (
        SELECT Volume
        FROM StockDetail
        WHERE StockSymbol = ?
        ORDER BY Date DESC
        LIMIT 30
      ) t
    `;
    const [avgRows] = await conn.query(avgVolSql, [normalizedSymbol]);
    const avgVolume30D = avgRows?.[0]?.AvgVolume30D
      ? Number(avgRows[0].AvgVolume30D)
      : 0;
    const formattedAvgVolume30D =
      avgVolume30D > 0 ? avgVolume30D.toFixed(2) : "0";

    // 6) ประวัติราคา (ตาม timeframe) — คงเดิม
    let historySql = `
      SELECT StockSymbol, Date, OpenPrice, HighPrice, LowPrice, ClosePrice
      FROM StockDetail
      WHERE StockSymbol = ?
      ORDER BY Date DESC
    `;
    const limit = historyLimits[timeframe];
    const params = [normalizedSymbol];
    if (limit !== null) {
      historySql += ` LIMIT ?`;
      params.push(limit);
    }
    const [historyRows] = await conn.query(historySql, params);
    const historicalPrices = [...historyRows].reverse();

    // 7) Overview — คงเดิม
    const peOverviewSql = `
      SELECT *
      FROM StockDetail
      WHERE StockSymbol = ?
        AND PERatio     IS NOT NULL
        AND MarketCap   IS NOT NULL
        AND TotalAssets IS NOT NULL
      ORDER BY Date DESC
      LIMIT 1
    `;
    const [peRows] = await conn.query(peOverviewSql, [normalizedSymbol]);
    const peOverview = peRows?.[0] || null;

    const overviewDefault = {
      ...stock,
      ClosePrice: closePriceNative,
      AvgVolume30D: formattedAvgVolume30D,
      ExchangeRateTHB: usdThb,
      IsRealtimePrice: livePrice != null,
    };
    const overview = peOverview || overviewDefault;

    const today = formatDateYYYYMMDD(new Date());

    return res.json({
      StockDetailID: stock.StockDetailID,
      StockSymbol: stock.StockSymbol,
      Type: stockType,
      company: stock.CompanyName,

      ClosePrice: closePriceNative,
      ClosePriceUSD: Number(Number(ClosePriceUSD).toFixed(2)),
      ClosePriceTHB: Number(Number(ClosePriceTHB).toFixed(2)),

      Date: today,
      Change: liveVsBaselinePct ?? null,
      Volume: stock.Volume,

      // === Baseline/Live ===
      BaselineClose: baselineClose,
      BaselineDate: baselineDate,
      BaselineSource: baselineSource,
      LivePrice: livePrice,
      LiveVsBaselinePct: liveVsBaselinePct,
      LiveVsBaselinePctAbs: liveVsBaselinePctAbs,
      PriceDelta: priceDelta,
      Direction: direction,
      IsRealtimePrice: livePrice != null,

      // Prediction
      PredictionClose: predictionClose,
      PredictionTrend: predictionTrend,
      PredictionCloseDate: stock.Date,
      PredictionClose_Ensemble: stock.PredictionClose_Ensemble,
      PricePredictionChange: pricePredictionChange,

      SelectedTimeframe: timeframe,
      HistoricalPrices: historicalPrices,

      Overview: overview,

      Profile: {
        Market: stock.Market,
        Sector: stock.Sector,
        Industry: stock.Industry,
        Description: stock.Description,
      },

      _tradingView: tv ? { price: toNum(tv?.price) } : null,
    });
  } catch (error) {
    console.error("Internal server error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

//Recommended US Stocks
app.get("/api/recommend-us-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุด
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res
          .status(500)
          .json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่ดึงหุ้น **Top 5 ของตลาด US**
      const recommendQuery = `
        SELECT 
          sd.StockDetailID, 
          s.StockSymbol, 
          sd.ClosePrice, 
          sd.Changepercen AS ChangePercentage
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ? AND s.Market = 'America'
        ORDER BY ABS(sd.Changepercen) DESC
        LIMIT 5;
      `;

      pool.query(recommendQuery, [latestDate], (recErr, recommendResults) => {
        if (recErr) {
          console.error("Database error fetching recommended stocks:", recErr);
          return res
            .status(500)
            .json({ error: "Database error fetching recommended stocks" });
        }

        const stockSymbols = recommendResults.map((stock) => stock.StockSymbol);

        // ดึงข้อมูลกราฟย้อนหลัง 5 วัน
        const historyQuery = `
          SELECT StockSymbol, Date, ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY StockSymbol, Date DESC
        `;

        pool.query(historyQuery, [stockSymbols], (histErr, historyResults) => {
          if (histErr) {
            console.error("Database error fetching historical data:", histErr);
            return res
              .status(500)
              .json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyResults.forEach((entry) => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice,
              });
            }
          });

          // ส่ง Response
          res.json({
            date: latestDate,
            recommendedStocks: recommendResults.map((stock) => ({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              ClosePrice: stock.ClosePrice,
              Change: stock.ChangePercentage,
              HistoricalPrices: historyMap[stock.StockSymbol] || [],
            })),
          });
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Most Held US Stocks
app.get("/api/most-held-us-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุด
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res
          .status(500)
          .json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่ดึงหุ้นทั้งหมดของตลาด US
      const mostHeldQuery = `
        SELECT 
          sd.StockDetailID,
          s.StockSymbol, 
          s.Market, 
          sd.ClosePrice, 
          sd.Changepercen AS ChangePercentage
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ? AND s.Market = 'America'
        ORDER BY s.StockSymbol ASC;
      `;

      pool.query(
        mostHeldQuery,
        [latestDate],
        (mostHeldErr, mostHeldResults) => {
          if (mostHeldErr) {
            console.error(
              "Database error fetching most held stocks:",
              mostHeldErr
            );
            return res
              .status(500)
              .json({ error: "Database error fetching most held stocks" });
          }

          const stockSymbols = mostHeldResults.map(
            (stock) => stock.StockSymbol
          );

          // ดึงข้อมูลกราฟย้อนหลัง 5 วัน
          const historyQuery = `
          SELECT StockSymbol, Date, ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY StockSymbol, Date DESC;
        `;

          pool.query(
            historyQuery,
            [stockSymbols],
            (histErr, historyResults) => {
              if (histErr) {
                console.error(
                  "Database error fetching historical data:",
                  histErr
                );
                return res
                  .status(500)
                  .json({ error: "Database error fetching historical data" });
              }

              // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
              const historyMap = {};
              historyResults.forEach((entry) => {
                if (!historyMap[entry.StockSymbol]) {
                  historyMap[entry.StockSymbol] = [];
                }
                if (historyMap[entry.StockSymbol].length < 5) {
                  historyMap[entry.StockSymbol].push({
                    Date: entry.Date,
                    ClosePrice: entry.ClosePrice,
                  });
                }
              });

              // ส่ง Response
              res.json({
                date: latestDate,
                mostHeldStocks: mostHeldResults.map((stock) => ({
                  StockDetailID: stock.StockDetailID,
                  StockSymbol: stock.StockSymbol,
                  Type: "US Stock",
                  ClosePrice: stock.ClosePrice,
                  Change: stock.ChangePercentage,
                  HistoricalPrices: historyMap[stock.StockSymbol] || [],
                })),
              });
            }
          );
        }
      );
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Recommended TH Stocks
app.get("/api/recommend-th-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุด
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res
          .status(500)
          .json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่ดึงหุ้น **Top 5 ของตลาดไทย**
      const recommendQuery = `
        SELECT 
          sd.StockDetailID, 
          s.StockSymbol, 
          sd.ClosePrice, 
          sd.Changepercen AS ChangePercentage
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ? AND s.Market = 'Thailand'
        ORDER BY ABS(sd.Changepercen) DESC
        LIMIT 5;
      `;

      pool.query(recommendQuery, [latestDate], (recErr, recommendResults) => {
        if (recErr) {
          console.error(
            "Database error fetching recommended Thai stocks:",
            recErr
          );
          return res
            .status(500)
            .json({ error: "Database error fetching recommended Thai stocks" });
        }

        const stockSymbols = recommendResults.map((stock) => stock.StockSymbol);

        // ดึงข้อมูลกราฟย้อนหลัง 5 วัน
        const historyQuery = `
          SELECT StockSymbol, Date, ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY StockSymbol, Date DESC
        `;

        pool.query(historyQuery, [stockSymbols], (histErr, historyResults) => {
          if (histErr) {
            console.error("Database error fetching historical data:", histErr);
            return res
              .status(500)
              .json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyResults.forEach((entry) => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice,
              });
            }
          });

          // ส่ง Response
          res.json({
            date: latestDate,
            recommendedStocks: recommendResults.map((stock) => ({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              ClosePrice: stock.ClosePrice,
              Change: stock.ChangePercentage,
              HistoricalPrices: historyMap[stock.StockSymbol] || [],
            })),
          });
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//Most Held Thai Stocks
app.get("/api/most-held-th-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุด
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res
          .status(500)
          .json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่ดึงหุ้นทั้งหมดของตลาดไทย
      const mostHeldQuery = `
        SELECT 
          sd.StockDetailID,
          s.StockSymbol, 
          s.Market, 
          sd.ClosePrice, 
          sd.Changepercen AS ChangePercentage
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ? AND s.Market = 'Thailand'
        ORDER BY s.StockSymbol ASC;
      `;

      pool.query(
        mostHeldQuery,
        [latestDate],
        (mostHeldErr, mostHeldResults) => {
          if (mostHeldErr) {
            console.error(
              "Database error fetching most held Thai stocks:",
              mostHeldErr
            );
            return res
              .status(500)
              .json({ error: "Database error fetching most held Thai stocks" });
          }

          const stockSymbols = mostHeldResults.map(
            (stock) => stock.StockSymbol
          );

          // ดึงข้อมูลกราฟย้อนหลัง 5 วัน
          const historyQuery = `
          SELECT StockSymbol, Date, ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY StockSymbol, Date DESC;
        `;

          pool.query(
            historyQuery,
            [stockSymbols],
            (histErr, historyResults) => {
              if (histErr) {
                console.error(
                  "Database error fetching historical data:",
                  histErr
                );
                return res
                  .status(500)
                  .json({ error: "Database error fetching historical data" });
              }

              // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
              const historyMap = {};
              historyResults.forEach((entry) => {
                if (!historyMap[entry.StockSymbol]) {
                  historyMap[entry.StockSymbol] = [];
                }
                if (historyMap[entry.StockSymbol].length < 5) {
                  historyMap[entry.StockSymbol].push({
                    Date: entry.Date,
                    ClosePrice: entry.ClosePrice,
                  });
                }
              });

              // ส่ง Response
              res.json({
                date: latestDate,
                mostHeldStocks: mostHeldResults.map((stock) => ({
                  StockDetailID: stock.StockDetailID,
                  StockSymbol: stock.StockSymbol,
                  Type: "TH Stock",
                  ClosePrice: stock.ClosePrice,
                  Change: stock.ChangePercentage,
                  HistoricalPrices: historyMap[stock.StockSymbol] || [],
                })),
              });
            }
          );
        }
      );
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

const API_KEY = process.env.FINNHUB_API_KEY;
const cheerio = require("cheerio");
const { timeStamp } = require("console");
async function getTradingViewPrice(symbol, market = "thailand", retries = 3) {
  const marketConfig = {
    thailand: {
      endpoint: "https://scanner.tradingview.com/thailand/scan",
      prefixes: ["SET:"],
    },
    usa: {
      endpoint: "https://scanner.tradingview.com/america/scan",
      prefixes: ["NASDAQ:", "NYSE:"],
    },
  };

  if (!marketConfig[market.toLowerCase()]) {
    throw new Error(`ตลาด '${market}' ไม่ได้รับการสนับสนุน`);
  }

  const { endpoint, prefixes } = marketConfig[market.toLowerCase()];
  let lastError;

  for (const prefix of prefixes) {
    const ticker = `${prefix}${symbol.toUpperCase()}`;
    const payload = {
      symbols: {
        tickers: [ticker],
        query: { types: [] },
      },
      columns: ["close", "description", "name"],
    };

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await axios.post(endpoint, payload, {
          headers: {
            "Content-Type": "application/json",
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", // เพิ่ม User-Agent
          },
          timeout: 5000, // Timeout 5 วินาที
        });

        console.log(
          `พยายามครั้งที่ ${attempt} - การตอบสนอง API สำหรับ ${ticker} ใน ${market}:`,
          response.data
        );

        const result = response.data?.data?.[0];
        if (!result) {
          throw new Error(
            `ไม่พบราคาหุ้น ${symbol} (ticker: ${ticker}) ในตลาด ${market}`
          );
        }

        return {
          symbol: result.d[2],
          name: result.d[1],
          price: result.d[0],
        };
      } catch (error) {
        console.error(
          `พยายามครั้งที่ ${attempt} ล้มเหลวสำหรับ ${ticker} ใน ${market}:`,
          error.message
        );
        lastError = error;
        if (attempt < retries) {
          await new Promise((resolve) => setTimeout(resolve, 1000 * attempt)); // Exponential backoff
        }
      }
    }
  }

  throw new Error(
    `ไม่สามารถดึงราคาหุ้น ${symbol} ใน ${market} ได้: ${lastError.message}`
  );
}

app.get("/api/price/:market/:symbol", async (req, res) => {
  const { symbol, market } = req.params;

  try {
    const data = await getTradingViewPrice(symbol, market);
    res.json({
      symbol: data.symbol,
      name: data.name,
      price: data.price,
      market: market.toLowerCase(),
    });
  } catch (e) {
    res.status(500).json({ detail: "ไม่สามารถดึงราคาได้: " + e.message });
  }
});

// Helper function to get THB to USD exchange rate
async function getThbToUsdRate() {
  try {
    const response = await axios.get(
      `https://api.exchangerate-api.com/v4/latest/THB`
    );
    // Return the rate for 1 THB to USD
    return response.data.rates.USD || 1 / 36.5;
  } catch (error) {
    console.error("Error fetching THB to USD exchange rate:", error);
    return 1 / 36.5; // Fallback
  }
}

// Helper function to check market status
function getMarketStatus(market) {
  const now = new Date();
  // // US Market (ET, UTC-4 for EDT)
  // if (market === 'America') {
  //     const nowET = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  //     const day = nowET.getDay(); // 0=Sun, 6=Sat
  //     const hour = nowET.getHours();
  //     const minute = nowET.getMinutes();

  //     if (day >= 1 && day <= 5) { // Mon-Fri
  //         if ((hour > 9 || (hour === 9 && minute >= 30)) && hour < 16) {
  //             return 'OPEN';
  //         }
  //     }
  //     return 'CLOSED';
  // }
  // // Thai Market (ICT, UTC+7)
  // if (market === 'Thailand') {
  //     const nowICT = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Bangkok' }));
  //     const day = nowICT.getDay();
  //     const hour = nowICT.getHours();
  //     const minute = nowICT.getMinutes();

  //     if (day >= 1 && day <= 5) { // Mon-Fri
  //         const isMorningSession = (hour >= 10 && (hour < 12 || (hour === 12 && minute <= 30)));
  //         const isAfternoonSession = ((hour > 14 || (hour === 14 && minute >= 30)) && (hour < 16 || (hour === 16 && minute <= 30)));
  //         if (isMorningSession || isAfternoonSession) {
  //             return 'OPEN';
  //         }
  //     }
  //     return 'CLOSED';
  // }
  // return 'UNKNOWN';
}

// API สำหรับดึงข้อมูล Portfolio ของผู้ใช้ พร้อมคำนวณกำไร/ขาดทุน
app.get("/api/portfolio", verifyToken, async (req, res) => {
  let connection;
  try {
    connection = await pool.promise().getConnection();

    // === helpers ===
    // % เปลี่ยนแปลงจาก A(base) -> B(live)
    function percentChange(live, base) {
      const L = Number(live),
        B = Number(base);
      if (!Number.isFinite(L) || !Number.isFinite(B) || B <= 0) return null;
      return ((L - B) / B) * 100;
    }

    // อัตราแปลง THB -> USD (เช่น 0.027xx)
    const thbToUsdRate = await getThbToUsdRate();

    // 1) ดึง portfolio
    const [portfolioRows] = await connection.query(
      "SELECT * FROM papertradeportfolio WHERE UserID = ?",
      [req.userId]
    );
    if (portfolioRows.length === 0) {
      return res
        .status(404)
        .json({ message: "ไม่พบ Portfolio สำหรับผู้ใช้นี้" });
    }
    const portfolio = portfolioRows[0];

    // 2) ดึง holdings (BuyPrice เก็บเป็น USD แล้ว)
    const [holdingsRows] = await connection.query(
      `SELECT 
         h.PaperHoldingID, 
         h.StockSymbol, 
         h.Quantity, 
         h.BuyPrice,     -- USD
         s.Market
       FROM paperportfolioholdings h
       JOIN Stock s ON h.StockSymbol = s.StockSymbol
       WHERE h.PaperPortfolioID = ?`,
      [portfolio.PaperPortfolioID]
    );

    // 3) ดึง "ราคาปัจจุบัน" จาก TradingView (จะใช้ค่านี้ในการคิด P/L และ % จาก BuyPrice)
    const pricePromises = holdingsRows.map(async (h) => {
      const tvMarket = h.Market === "Thailand" ? "thailand" : "usa";
      try {
        const p = await getTradingViewPrice(h.StockSymbol, tvMarket);
        const native = Number(p?.price);
        const currentPriceNative = Number.isFinite(native) ? native : 0;

        // แปลงเป็น USD เสมอ (TH → คูณ USD/THB, US → คงเดิม)
        const currentPriceUSD =
          h.Market === "Thailand"
            ? currentPriceNative * thbToUsdRate
            : currentPriceNative;

        return { symbol: h.StockSymbol, currentPriceUSD };
      } catch (e) {
        console.error(`TradingView error ${h.StockSymbol}:`, e?.message || e);
        return { symbol: h.StockSymbol, currentPriceUSD: 0, error: true };
      }
    });

    const priceList = await Promise.all(pricePromises);
    const priceMapUSD = priceList.reduce((m, x) => {
      m[x.symbol] = x.currentPriceUSD;
      return m;
    }, {});

    // 4) รวม Lot เดียวกัน (คำนวณ cost basis รวม)
    const grouped = holdingsRows.reduce((acc, h) => {
      if (!acc[h.StockSymbol]) {
        acc[h.StockSymbol] = {
          StockSymbol: h.StockSymbol,
          Market: h.Market,
          TotalQuantity: 0,
          TotalCostBasisUSD: 0, // BuyPrice เป็น USD อยู่แล้ว
        };
      }
      const qty = Number(h.Quantity) || 0;
      const buyUSD = Number(h.BuyPrice) || 0;
      acc[h.StockSymbol].TotalQuantity += qty;
      acc[h.StockSymbol].TotalCostBasisUSD += buyUSD * qty;
      return acc;
    }, {});

    // 5) คำนวณมูลค่า & P/L และ "เปอร์เซ็นต์ขึ้นลงจากราคาเฉลี่ยซื้อ" ด้วยราคา API ปัจจุบัน
    let totalHoldingsValueUSD = 0;

    const holdingsWithPL = Object.values(grouped).map((g) => {
      const qty = g.TotalQuantity || 0;
      const costUSD = g.TotalCostBasisUSD || 0;
      const curPriceUSD = Number(priceMapUSD[g.StockSymbol]) || 0;

      const currentValueUSD = curPriceUSD * qty;
      const avgBuyPriceUSD = qty > 0 ? costUSD / qty : 0;

      const unrealizedPL_USD = currentValueUSD - costUSD;
      const unrealizedPLPercent =
        costUSD > 0 ? (unrealizedPL_USD / costUSD) * 100 : 0;

      // ★ ใหม่: % ขึ้นลงจาก "ราคาที่ซื้อเฉลี่ย (USD)" เทียบ "ราคา API ปัจจุบัน (USD)"
      const priceChangeFromBuyPct = percentChange(curPriceUSD, avgBuyPriceUSD);
      const isUpFromBuy =
        priceChangeFromBuyPct != null ? priceChangeFromBuyPct > 0 : null;

      totalHoldingsValueUSD += currentValueUSD;

      return {
        StockSymbol: g.StockSymbol,
        Quantity: qty,

        AvgBuyPriceUSD: avgBuyPriceUSD.toFixed(2),
        CurrentPriceUSD: curPriceUSD.toFixed(2), // จาก TradingView (TH แปลง THB→USD แล้ว)
        CurrentValueUSD: currentValueUSD.toFixed(2),

        UnrealizedPL_USD: unrealizedPL_USD.toFixed(2),
        UnrealizedPLPercent: `${unrealizedPLPercent.toFixed(2)}%`,

        // ★ ฟิลด์ใหม่
        PriceChangeFromBuyPercent:
          priceChangeFromBuyPct != null
            ? `${priceChangeFromBuyPct.toFixed(2)}%`
            : "-",
        IsPriceAboveBuy: isUpFromBuy, // true=API > AvgBuy, false=ต่ำกว่า, null=คำนวณไม่ได้

        Market: g.Market,
        MarketStatus: getMarketStatus(g.Market),
        PriceSource: "TradingView",
      };
    });

    // 6) รวมพอร์ต
    const balanceUSD = Number(portfolio.Balance) || 0;
    portfolio.BalanceUSD = balanceUSD.toFixed(2);
    portfolio.TotalHoldingsValueUSD = Number(totalHoldingsValueUSD.toFixed(2));
    portfolio.TotalPortfolioValueUSD = Number(
      (balanceUSD + totalHoldingsValueUSD).toFixed(2)
    );
    portfolio.holdings = holdingsWithPL;

    // (ออปชัน) รวม P/L ทั้งพอร์ต
    const totalCostUSD = Object.values(grouped).reduce(
      (sum, g) => sum + g.TotalCostBasisUSD,
      0
    );
    const totalUnrealizedPL_USD = totalHoldingsValueUSD - totalCostUSD;
    const totalUnrealizedPLPercent =
      totalCostUSD > 0 ? (totalUnrealizedPL_USD / totalCostUSD) * 100 : 0;

    portfolio.TotalUnrealizedPL_USD = Number(totalUnrealizedPL_USD.toFixed(2));
    portfolio.TotalUnrealizedPLPercent = `${totalUnrealizedPLPercent.toFixed(
      2
    )}%`;

    res
      .status(200)
      .json({ message: "ดึงข้อมูล Portfolio สำเร็จ", data: portfolio });
  } catch (error) {
    console.error("Error fetching portfolio:", error);
    res.status(500).json({ error: "Internal server error" });
  } finally {
    if (connection) connection.release();
  }
});

// POST /api/autoTrade/run

// app.post("/api/autoTrade/run", async (req, res) => {
//   let connection;

//   // ====== CONFIG ======
//   const PORTFOLIO_CCY = "USD";
//   const DEFAULT_LIMIT = 50;
//   const HARD_CAP_LIMIT = 200;
//   const EFFECTIVE_LIMIT = Math.min(DEFAULT_LIMIT, HARD_CAP_LIMIT);
//   const CONCURRENCY = 8;
//   const FEE_RATE = 0.1 / 100; // 0.15%

//   // ====== HELPERS ======
//   const toMoney2 = (n) => Number(Number(n || 0).toFixed(2)); // เงิน 2 ตำแหน่ง (USD)
//   const to6 = (n) => Number(Number(n || 0).toFixed(6)); // ราคา/สัดส่วน 6 ตำแหน่ง
//   const normSymbol = (s) =>
//     String(s || "")
//       .toUpperCase()
//       .slice(0, 10);

//   // --- ระบุว่าเป็นหุ้นไทยไหม: Market=THAILAND หรือสัญลักษณ์ .BK
//   const isThaiMarket = (marketRaw = "", rawSym = "") =>
//     /^THAILAND$/i.test(String(marketRaw).trim()) ||
//     /\.BK$/i.test(String(rawSym));

//   // --- แปลง market สำหรับ TradingView (แล้วแต่ฟังก์ชัน getTradingViewPrice ของคุณรองรับ)
//   // ถ้าเดิมคุณใช้ "usa" กับ "thailand" ก็ปรับตามนั้น
//   const toTvMarket = (marketRaw = "", rawSym = "") =>
//     isThaiMarket(marketRaw, rawSym) ? "thailand" : "usa";

//   // upsert holdings: เก็บ BuyPrice เป็น USD เสมอ (Type='AUTO')
//   async function upsertHolding(
//     conn,
//     { portfolioId, symbol, buyQty, priceUSD }
//   ) {
//     const [rows] = await conn.query(
//       `SELECT PaperHoldingID, Quantity, BuyPrice
//          FROM paperportfolioholdings
//         WHERE PaperPortfolioID = ? AND StockSymbol = ?
//         FOR UPDATE`,
//       [portfolioId, symbol]
//     );

//     const p = to6(priceUSD); // ราคา USD/หุ้น ที่จะบันทึก

//     if (rows?.length) {
//       const { PaperHoldingID, Quantity: oldQty, BuyPrice: oldPrice } = rows[0];
//       const newQty = Number(oldQty) + Number(buyQty);
//       const newBuyPrice =
//         newQty > 0
//           ? to6(
//               (Number(oldQty) * Number(oldPrice) + Number(buyQty) * p) / newQty
//             )
//           : p;

//       await conn.query(
//         `UPDATE paperportfolioholdings
//             SET Quantity = ?, BuyPrice = ?, Type = 'AUTO'
//           WHERE PaperHoldingID = ?`,
//         [newQty, newBuyPrice, PaperHoldingID]
//       );
//       return {
//         mode: "update",
//         newQty,
//         newBuyPrice,
//         paperHoldingId: PaperHoldingID,
//       };
//     } else {
//       const [ins] = await conn.query(
//         `INSERT INTO paperportfolioholdings
//            (Quantity, BuyPrice, PaperPortfolioID, StockSymbol, Type)
//          VALUES (?, ?, ?, ?, 'AUTO')`,
//         [buyQty, p, portfolioId, symbol]
//       );
//       return {
//         mode: "insert",
//         newQty: buyQty,
//         newBuyPrice: p,
//         paperHoldingId: ins.insertId,
//       };
//     }
//   }

//   // concurrency helper
//   const mapLimit = async (items, limit, worker) => {
//     const out = new Array(items.length);
//     let i = 0;
//     const runners = Array(Math.min(limit, items.length))
//       .fill(0)
//       .map(async () => {
//         while (true) {
//           const idx = i++;
//           if (idx >= items.length) break;
//           out[idx] = await worker(items[idx], idx);
//         }
//       });
//     await Promise.all(runners);
//     return out;
//   };

//   try {
//     connection = await pool.promise().getConnection();

//     // 1) เลือกพอร์ตที่เปิด ON
//     const [portfolios] = await connection.query(
//       `SELECT PaperPortfolioID, UserID, Balance, autoBalance, status, leverage
//          FROM papertradeportfolio
//         WHERE LOWER(status) = 'on'`
//     );
//     if (!portfolios?.length) {
//       return res.json({
//         status: "SKIPPED",
//         reason: "ไม่มีพอร์ตที่เปิดสถานะ ON",
//       });
//     }

//     // 2) ดึงราคาปิดล่าสุด + prediction (ของไทยเป็น THB)
//     const [priceRows] = await connection.query(
//       `SELECT
//         s.StockSymbol,
//         s.Market,
//         sd.ClosePrice,
//         sd.PredictionClose_Ensemble,
//         sd.Date AS DbPriceDate,
//         sd.StockDetailID
//       FROM stock s
//       JOIN (
//         SELECT sd1.StockSymbol, sd1.ClosePrice, sd1.PredictionClose_Ensemble, sd1.Date, sd1.StockDetailID
//         FROM stockdetail sd1
//         JOIN (
//           SELECT StockSymbol, MAX(Date) AS MaxDate
//           FROM stockdetail
//           GROUP BY StockSymbol
//         ) m ON m.StockSymbol = sd1.StockSymbol AND m.MaxDate = sd1.Date
//       ) sd ON sd.StockSymbol = s.StockSymbol
//       ORDER BY s.StockSymbol
//       LIMIT ?`,
//       [EFFECTIVE_LIMIT]
//     );
//     if (!priceRows?.length) {
//       return res
//         .status(404)
//         .json({ message: "ไม่พบข้อมูลหุ้นหรือราคาปิดล่าสุด" });
//     }

//     // 2.1 เตรียมเรต THB->USD แค่ครั้งเดียวให้ทั้งรอบ เพื่อประหยัด call (ยังคงใช้ฟังก์ชันด้านบน)
//     // หมายเหตุ: ถ้าไม่มีหุ้นไทยเลยก็ไม่โดนใช้
//     let thbToUsdRate = null;
//     const needThbRate = priceRows.some((r) =>
//       isThaiMarket(r.Market, r.StockSymbol)
//     );
//     if (needThbRate) {
//       thbToUsdRate = await getThbToUsdRate(); // ค่า "USD ต่อ 1 THB"
//     }

//     // 3) Enrich: ดึงราคา realtime จาก TradingView
//     const enriched = await mapLimit(priceRows, CONCURRENCY, async (row) => {
//       const rawSym = String(row.StockSymbol || "");
//       const symbol = normSymbol(rawSym.replace(/\.BK$/i, ""));
//       const marketRaw = row.Market || "";
//       const tvMarket = toTvMarket(marketRaw, rawSym);
//       const thai = isThaiMarket(marketRaw, rawSym);

//       // ดึงราคา realtime: ถ้าไทย -> THB, ถ้า US -> USD
//       let apiPrice = null; // ณสกุลท้องถิ่น
//       let apiError = null;
//       try {
//         const r = await getTradingViewPrice(symbol, tvMarket);
//         const p = Number(r?.price);
//         if (Number.isFinite(p) && p > 0) apiPrice = p;
//         else apiError = "invalid_price";
//       } catch (e) {
//         apiError = e?.message || "api_error";
//       }

//       // แปลงราคา/พยากรณ์เป็น USD เฉพาะหุ้นไทย
//       const fx = thai ? Number(thbToUsdRate || 1 / 36.5) : 1;

//       const priceLocal = apiPrice != null ? to6(apiPrice) : null; // THB (ไทย) / USD (US)
//       const priceUSD = apiPrice != null ? to6(apiPrice * fx) : null; // ถ้าไทย: THB*rate -> USD, ถ้า US: *1

//       const predRaw = Number(row.PredictionClose_Ensemble); // THB ถ้าไทย / USD ถ้า US
//       const predictionLocal = Number.isFinite(predRaw) ? to6(predRaw) : null;
//       const predictionUSD = Number.isFinite(predRaw) ? to6(predRaw * fx) : null;

//       let score = null; // ใช้ USD เท่านั้น
//       if (priceUSD != null && priceUSD > 0 && predictionUSD != null) {
//         score = to6((predictionUSD - priceUSD) / priceUSD);
//       }

//       return {
//         stockSymbol: symbol,
//         market: marketRaw,
//         stockDetailId: row.StockDetailID,
//         isThai: thai,
//         fxRate: to6(fx),
//         apiError,

//         // สำหรับโชว์
//         priceLocal, // THB ถ้าไทย / USD ถ้า US
//         predictionLocal, // THB ถ้าไทย / USD ถ้า US

//         // สำหรับคำนวณจริง
//         priceUSD, // ✅ ราคา USD/หุ้น
//         predictionUSD, // ✅ พยากรณ์ USD/หุ้น
//         score,
//       };
//     });

//     const results = [];

//     // 4) ทำทีละพอร์ต (transaction แยก)
//     for (const pf of portfolios) {
//       const {
//         PaperPortfolioID: portfolioId,
//         UserID: userId,
//         Balance: rawBalance,
//         autoBalance: rawAutoBalance,
//         leverage: rawAveragePct,
//       } = pf;

//       const balance = Number(rawBalance) || 0;
//       const riskPct = Math.max(0, Number(rawAveragePct) || 0);
//       let autoBudget = Math.max(0, Number(rawAutoBalance) || 0);

//       await connection.beginTransaction();

//       // โหลด holdings AUTO และล็อก
//       const [holdRows] = await connection.query(
//         `SELECT PaperHoldingID, StockSymbol, Quantity, BuyPrice
//            FROM paperportfolioholdings
//           WHERE PaperPortfolioID = ? AND Type = 'AUTO'
//           FOR UPDATE`,
//         [portfolioId]
//       );
//       const holdingsBySymbol = new Map(
//         (holdRows || []).map((r) => [normSymbol(r.StockSymbol), r])
//       );

//       // ===== SELL: ถืออยู่ + score < 0 => ขาย, เติมงบด้วย "สุทธิ" USD =====
//       const sellCandidates = enriched
//         .filter(
//           (x) =>
//             (x.score ?? 0) < 0 &&
//             holdingsBySymbol.has(x.stockSymbol) &&
//             Number.isFinite(x.priceUSD)
//         )
//         .sort((a, b) => (a.score ?? 0) - (b.score ?? 0));

//       const sellOrders = [];
//       let sellProceeds = 0;

//       for (const it of sellCandidates) {
//         const h = holdingsBySymbol.get(it.stockSymbol);
//         const qty = Math.floor(Number(h?.Quantity) || 0);
//         if (qty <= 0) continue;

//         const grossProceedUSD = qty * it.priceUSD;
//         const feeUSD = grossProceedUSD * FEE_RATE;
//         const netProceedUSD = toMoney2(grossProceedUSD - feeUSD); // ขาย = ราคา - ค่าธรรมเนียม

//         await connection.query(
//           `INSERT INTO papertrade
//              (PaperPortfolioID, TradeType, Quantity, Price, TradeDate, UserID, StockSymbol)
//            VALUES (?, 'sell', ?, ?, NOW(), ?, ?)`,
//           [portfolioId, qty, netProceedUSD, userId, normSymbol(it.stockSymbol)]
//         );

//         const remain = (Number(h.Quantity) || 0) - qty;
//         if (remain > 0) {
//           await connection.query(
//             `UPDATE paperportfolioholdings SET Quantity = ? WHERE PaperHoldingID = ?`,
//             [remain, h.PaperHoldingID]
//           );
//         } else {
//           await connection.query(
//             `DELETE FROM paperportfolioholdings WHERE PaperHoldingID = ?`,
//             [h.PaperHoldingID]
//           );
//         }

//         sellProceeds = toMoney2(sellProceeds + netProceedUSD);

//         sellOrders.push({
//           stockSymbol: normSymbol(it.stockSymbol),
//           quantity: qty,
//           priceLocal: it.priceLocal,
//           priceUSD: it.priceUSD,
//           predictionLocal: it.predictionLocal,
//           predictionUSD: it.predictionUSD,
//           grossProceedUSD: toMoney2(grossProceedUSD),
//           feeUSD: toMoney2(feeUSD),
//           netProceedUSD,
//           score: it.score,
//           stockDetailId: it.stockDetailId,
//           market: it.market,
//           isThai: it.isThai,
//           fxRate: it.fxRate,
//         });
//       }

//       if (sellProceeds > 0) {
//         await connection.query(
//           `UPDATE papertradeportfolio
//               SET autoBalance = autoBalance + ?
//             WHERE PaperPortfolioID = ? AND UserID = ?`,
//           [toMoney2(sellProceeds), portfolioId, userId]
//         );
//         autoBudget = toMoney2(autoBudget + sellProceeds);
//       }

//       // ===== BUY: score > 0, จัดงบตามน้ำหนัก score, ซื้อ = ราคา + ค่าธรรมเนียม =====
//       const buyList = enriched
//         .filter((x) => (x.score ?? 0) > 0 && Number.isFinite(x.priceUSD))
//         .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

//       const scoreSum = buyList.reduce(
//         (s, x) => s + Math.max(0, x.score || 0),
//         0
//       );
//       const equalWeight = buyList.length ? 1 / buyList.length : 0;

//       const targetBudgets = buyList.map((x) => {
//         const w =
//           scoreSum > 0 ? Math.max(0, x.score || 0) / scoreSum : equalWeight;
//         const budget = toMoney2(autoBudget * w);
//         return { sym: normSymbol(x.stockSymbol), w, budget };
//       });

//       let remainingBudget = autoBudget;
//       const buyOrders = [];
//       let usedAmount = 0;

//       // รอบแรก: ใช้งบตามน้ำหนัก
//       for (let i = 0; i < buyList.length; i++) {
//         const it = buyList[i];
//         const tgt = targetBudgets[i];
//         if (remainingBudget <= 0) break;

//         const allowed = Math.min(tgt.budget, remainingBudget);
//         const qty = Math.floor(allowed / it.priceUSD); // ใช้ราคา USD
//         if (qty <= 0) continue;

//         const grossCostUSD = qty * it.priceUSD;
//         const feeUSD = grossCostUSD * FEE_RATE;
//         const totalSpendUSD = toMoney2(grossCostUSD + feeUSD); // ซื้อ = ราคา + ค่าธรรมเนียม

//         await connection.query(
//           `INSERT INTO papertrade
//              (PaperPortfolioID, TradeType, Quantity, Price, TradeDate, UserID, StockSymbol)
//            VALUES (?, 'buy', ?, ?, NOW(), ?, ?)`,
//           [portfolioId, qty, totalSpendUSD, userId, normSymbol(it.stockSymbol)]
//         );

//         await upsertHolding(connection, {
//           portfolioId,
//           symbol: normSymbol(it.stockSymbol),
//           buyQty: qty,
//           priceUSD: it.priceUSD, // ✅ เก็บ BuyPrice เป็น USD/หุ้น เสมอ
//         });

//         buyOrders.push({
//           stockSymbol: normSymbol(it.stockSymbol),
//           quantity: qty,
//           priceLocal: it.priceLocal,
//           priceUSD: it.priceUSD,
//           predictionLocal: it.predictionLocal,
//           predictionUSD: it.predictionUSD,
//           grossCostUSD: toMoney2(grossCostUSD),
//           feeUSD: toMoney2(feeUSD),
//           totalSpendUSD,
//           score: it.score,
//           stockDetailId: it.stockDetailId,
//           market: it.market,
//           isThai: it.isThai,
//           fxRate: it.fxRate,
//         });

//         remainingBudget = toMoney2(remainingBudget - totalSpendUSD);
//         usedAmount = toMoney2(usedAmount + totalSpendUSD);
//       }

//       // รอบสอง: เก็บเศษงบ
//       if (remainingBudget > 0) {
//         for (const it of buyList) {
//           if (remainingBudget <= 0) break;

//           const qty = Math.floor(remainingBudget / it.priceUSD);
//           if (qty <= 0) continue;

//           const grossCostUSD = qty * it.priceUSD;
//           const feeUSD = grossCostUSD * FEE_RATE;
//           const totalSpendUSD = toMoney2(grossCostUSD + feeUSD);

//           await connection.query(
//             `INSERT INTO papertrade
//                (PaperPortfolioID, TradeType, Quantity, Price, TradeDate, UserID, StockSymbol)
//              VALUES (?, 'buy', ?, ?, NOW(), ?, ?)`,
//             [
//               portfolioId,
//               qty,
//               totalSpendUSD,
//               userId,
//               normSymbol(it.stockSymbol),
//             ]
//           );

//           await upsertHolding(connection, {
//             portfolioId,
//             symbol: normSymbol(it.stockSymbol),
//             buyQty: qty,
//             priceUSD: it.priceUSD,
//           });

//           buyOrders.push({
//             stockSymbol: normSymbol(it.stockSymbol),
//             quantity: qty,
//             priceLocal: it.priceLocal,
//             priceUSD: it.priceUSD,
//             predictionLocal: it.predictionLocal,
//             predictionUSD: it.predictionUSD,
//             grossCostUSD: toMoney2(grossCostUSD),
//             feeUSD: toMoney2(feeUSD),
//             totalSpendUSD,
//             score: it.score,
//             stockDetailId: it.stockDetailId,
//             market: it.market,
//             isThai: it.isThai,
//             fxRate: it.fxRate,
//           });

//           remainingBudget = toMoney2(remainingBudget - totalSpendUSD);
//           usedAmount = toMoney2(usedAmount + totalSpendUSD);
//         }
//       }

//       // หัก autoBalance ด้วย “ที่จ่ายจริงรวมค่าธรรมเนียม”
//       if (usedAmount > 0) {
//         const [upd] = await connection.query(
//           `UPDATE papertradeportfolio
//               SET autoBalance = GREATEST(0, autoBalance - ?)
//             WHERE PaperPortfolioID = ? AND UserID = ?`,
//           [toMoney2(usedAmount), portfolioId, userId]
//         );
//         if (!upd?.affectedRows) {
//           await connection.rollback();
//           results.push({
//             portfolioId,
//             userId,
//             status: "FAILED",
//             error: "ปรับปรุง autoBalance ไม่สำเร็จ",
//           });
//           continue;
//         }
//       }

//       await connection.commit();

//       results.push({
//         status: "EXECUTED",
//         userId,
//         portfolio: {
//           paperPortfolioId: portfolioId,
//           balance,
//           autoBalanceBefore: toMoney2(rawAutoBalance),
//           buysCostUSD: toMoney2(usedAmount),
//           autoBalanceAfter: toMoney2(
//             (Number(rawAutoBalance) || 0) - usedAmount
//           ),
//           average: riskPct,
//           currency: PORTFOLIO_CCY,
//         },
//         counts: {
//           consideredSymbols: enriched.length,
//           sellPlaced: sellOrders.length,
//           buyPlaced: buyOrders.length,
//         },
//         orders: { sells: sellOrders, buys: buyOrders },
//       });
//     } // end portfolios loop

//     // ✅ รายการ FX ของหุ้นไทยหลังแปลงแล้ว (debug/inspect)
//     const thaiConverted = enriched
//       .filter((x) => x.isThai)
//       .map((x) => ({
//         stockSymbol: x.stockSymbol,
//         fxRateTHB2USD: x.fxRate,
//         priceTHB: x.priceLocal,
//         priceUSD: x.priceUSD,
//         predictionTHB: x.predictionLocal,
//         predictionUSD: x.predictionUSD,
//         score: x.score,
//         stockDetailId: x.stockDetailId,
//       }));

//     // Response
//     return res.json({
//       status: "DONE",
//       note: "ใช้ getThbToUsdRate() สำหรับหุ้นไทยเท่านั้น: THB → USD ก่อนคำนวณทั้งหมด | SELL=ราคา−fee เติม autoBalance | BUY=ราคา+fee หัก autoBalance | BuyPrice เก็บ USD/หุ้น",
//       thaiConverted,
//       enrichedList: enriched.map((x) => ({
//         stockSymbol: x.stockSymbol,
//         market: x.market,
//         isThai: x.isThai,
//         fxRate: x.fxRate,
//         priceLocal: x.priceLocal,
//         priceUSD: x.priceUSD,
//         predictionLocal: x.predictionLocal,
//         predictionUSD: x.predictionUSD,
//         score: x.score,
//         stockDetailId: x.stockDetailId,
//       })),
//       portfolios: results,
//     });
//   } catch (err) {
//     try {
//       if (connection) await connection.rollback();
//     } catch {}
//     console.error("AutoTrade Error:", err);
//     return res
//       .status(500)
//       .json({ error: "เกิดข้อผิดพลาด", detail: err?.message || String(err) });
//   } finally {
//     if (connection) connection.release();
//   }
// });

app.post("/api/autoTrade/run", async (req, res) => {
  let connection;

  // ====== CONFIG ======
  const PORTFOLIO_CCY = "USD";
  const DEFAULT_LIMIT = 50;
  const HARD_CAP_LIMIT = 200;
  const EFFECTIVE_LIMIT = Math.min(DEFAULT_LIMIT, HARD_CAP_LIMIT);
  const CONCURRENCY = 8;
  const FEE_RATE = 0.1 / 100; // 0.1%

  // ====== HELPERS ======
  const toMoney2 = (n) => Number(Number(n || 0).toFixed(2)); // เงิน 2 ตำแหน่ง (USD)
  const to6 = (n) => Number(Number(n || 0).toFixed(6)); // ราคา/สัดส่วน 6 ตำแหน่ง
  const normSymbol = (s) =>
    String(s || "")
      .toUpperCase()
      .slice(0, 10);

  // หุ้นไทย? (Market=THAILAND หรือสัญลักษณ์ลงท้าย .BK)
  const isThaiMarket = (marketRaw = "", rawSym = "") =>
    /^THAILAND$/i.test(String(marketRaw).trim()) ||
    /\.BK$/i.test(String(rawSym));

  // map ตลาดไปยังค่าใช้กับ TradingView
  const toTvMarket = (marketRaw = "", rawSym = "") =>
    isThaiMarket(marketRaw, rawSym) ? "thailand" : "usa";

  // upsert holdings: เก็บ BuyPrice เป็น USD เสมอ (Type='AUTO')
  async function upsertHolding(
    conn,
    { portfolioId, symbol, buyQty, priceUSD }
  ) {
    const [rows] = await conn.query(
      `SELECT PaperHoldingID, Quantity, BuyPrice
         FROM paperportfolioholdings
        WHERE PaperPortfolioID = ? AND StockSymbol = ?
        FOR UPDATE`,
      [portfolioId, symbol]
    );

    const p = to6(priceUSD); // ราคา USD/หุ้น

    if (rows?.length) {
      const { PaperHoldingID, Quantity: oldQty, BuyPrice: oldPrice } = rows[0];
      const newQty = Number(oldQty) + Number(buyQty);
      const newBuyPrice =
        newQty > 0
          ? to6(
              (Number(oldQty) * Number(oldPrice) + Number(buyQty) * p) / newQty
            )
          : p;

      await conn.query(
        `UPDATE paperportfolioholdings
            SET Quantity = ?, BuyPrice = ?, Type = 'AUTO'
          WHERE PaperHoldingID = ?`,
        [newQty, newBuyPrice, PaperHoldingID]
      );
      return {
        mode: "update",
        newQty,
        newBuyPrice,
        paperHoldingId: PaperHoldingID,
      };
    } else {
      const [ins] = await conn.query(
        `INSERT INTO paperportfolioholdings
           (Quantity, BuyPrice, PaperPortfolioID, StockSymbol, Type)
         VALUES (?, ?, ?, ?, 'AUTO')`,
        [buyQty, p, portfolioId, symbol]
      );
      return {
        mode: "insert",
        newQty: buyQty,
        newBuyPrice: p,
        paperHoldingId: ins.insertId,
      };
    }
  }

  // concurrency helper
  const mapLimit = async (items, limit, worker) => {
    const out = new Array(items.length);
    let i = 0;
    const runners = Array(Math.min(limit, items.length))
      .fill(0)
      .map(async () => {
        while (true) {
          const idx = i++;
          if (idx >= items.length) break;
          out[idx] = await worker(items[idx], idx);
        }
      });
    await Promise.all(runners);
    return out;
  };

  try {
    connection = await pool.promise().getConnection();

    // 1) เลือกพอร์ตที่เปิด ON
    const [portfolios] = await connection.query(
      `SELECT PaperPortfolioID, UserID, Balance, autoBalance, status, leverage
         FROM papertradeportfolio
        WHERE LOWER(status) = 'on'`
    );
    if (!portfolios?.length) {
      return res.json({
        status: "SKIPPED",
        reason: "ไม่มีพอร์ตที่เปิดสถานะ ON",
      });
    }

    // 2) ดึงราคาปิดล่าสุด + prediction (ของไทยเป็น THB)
    const [priceRows] = await connection.query(
      `SELECT
        s.StockSymbol,
        s.Market,
        sd.ClosePrice,
        sd.PredictionClose_Ensemble,
        sd.Date AS DbPriceDate,
        sd.StockDetailID
      FROM stock s
      JOIN (
        SELECT sd1.StockSymbol, sd1.ClosePrice, sd1.PredictionClose_Ensemble, sd1.Date, sd1.StockDetailID
        FROM stockdetail sd1
        JOIN (
          SELECT StockSymbol, MAX(Date) AS MaxDate
          FROM stockdetail
          GROUP BY StockSymbol
        ) m ON m.StockSymbol = sd1.StockSymbol AND m.MaxDate = sd1.Date
      ) sd ON sd.StockSymbol = s.StockSymbol
      ORDER BY s.StockSymbol
      LIMIT ?`,
      [EFFECTIVE_LIMIT]
    );
    if (!priceRows?.length) {
      return res
        .status(404)
        .json({ message: "ไม่พบข้อมูลหุ้นหรือราคาปิดล่าสุด" });
    }

    // 2.1 เตรียมเรต THB->USD (ครั้งเดียวต่อรอบ)
    let thbToUsdRate = null;
    const needThbRate = priceRows.some((r) =>
      isThaiMarket(r.Market, r.StockSymbol)
    );
    if (needThbRate) {
      thbToUsdRate = await getThbToUsdRate(); // USD ต่อ 1 THB
    }

    // 3) Enrich: ดึงราคา realtime จาก TradingView (ไทยเป็น THB, US เป็น USD)
    const enriched = await mapLimit(priceRows, CONCURRENCY, async (row) => {
      const rawSym = String(row.StockSymbol || "");
      const symbol = normSymbol(rawSym.replace(/\.BK$/i, ""));
      const marketRaw = row.Market || "";
      const tvMarket = toTvMarket(marketRaw, rawSym);
      const thai = isThaiMarket(marketRaw, rawSym);

      let apiPrice = null; // ราคา ณ สกุลท้องถิ่น
      let apiError = null;
      try {
        const r = await getTradingViewPrice(symbol, tvMarket);
        const p = Number(r?.price);
        if (Number.isFinite(p) && p > 0) apiPrice = p;
        else apiError = "invalid_price";
      } catch (e) {
        apiError = e?.message || "api_error";
      }

      const fx = thai ? Number(thbToUsdRate || 1 / 36.5) : 1;

      const priceLocal = apiPrice != null ? to6(apiPrice) : null; // THB ถ้าไทย / USD ถ้า US
      const priceUSD = apiPrice != null ? to6(apiPrice * fx) : null;

      const predRaw = Number(row.PredictionClose_Ensemble);
      const predictionLocal = Number.isFinite(predRaw) ? to6(predRaw) : null;
      const predictionUSD = Number.isFinite(predRaw) ? to6(predRaw * fx) : null;

      let score = null; // ใช้ USD เท่านั้น
      if (priceUSD != null && priceUSD > 0 && predictionUSD != null) {
        score = to6((predictionUSD - priceUSD) / priceUSD);
      }

      return {
        stockSymbol: symbol,
        market: marketRaw,
        stockDetailId: row.StockDetailID,
        isThai: thai,
        fxRate: to6(fx),
        apiError,

        // สำหรับโชว์
        priceLocal,
        predictionLocal,

        // สำหรับคำนวณ
        priceUSD,
        predictionUSD,
        score,
      };
    });

    const results = [];

    // 4) ทำทีละพอร์ต (transaction แยก)
    for (const pf of portfolios) {
      const {
        PaperPortfolioID: portfolioId,
        UserID: userId,
        Balance: rawBalance,
        autoBalance: rawAutoBalance,
        leverage: rawLeveragePct,
      } = pf;

      const balance = Number(rawBalance) || 0;
      let autoBudget = Math.max(0, Number(rawAutoBalance) || 0);

      // แปลความ leverage = stop-loss %
      const stopLossPct = Math.max(
        0,
        Math.min(50, Number(rawLeveragePct) || 0)
      ); // clamp 0–50%
      const stopLossFrac = stopLossPct / 100;

      await connection.beginTransaction();

      // โหลด holdings AUTO และล็อก
      const [holdRows] = await connection.query(
        `SELECT PaperHoldingID, StockSymbol, Quantity, BuyPrice
           FROM paperportfolioholdings
          WHERE PaperPortfolioID = ? AND Type = 'AUTO'
          FOR UPDATE`,
        [portfolioId]
      );
      const holdingsBySymbol = new Map(
        (holdRows || []).map((r) => [normSymbol(r.StockSymbol), r])
      );

      // ===== SELL: (1) score < 0  หรือ  (2) หลุด stop-loss จาก BuyPrice (USD) =====
      const sellCandidates = enriched
        .filter((x) => {
          const h = holdingsBySymbol.get(x.stockSymbol);
          if (!h) return false;
          if (!Number.isFinite(x.priceUSD) || x.priceUSD <= 0) return false;

          const negScore = (x.score ?? 0) < 0;

          const buyPx = Number(h.BuyPrice);
          const hasStop =
            stopLossFrac > 0 && Number.isFinite(buyPx) && buyPx > 0;
          const drawdown = hasStop ? (x.priceUSD - buyPx) / buyPx : null; // <0 = ขาดทุน
          const hitStop = hasStop && drawdown <= -stopLossFrac;

          return negScore || hitStop;
        })
        .sort((a, b) => (a.score ?? 0) - (b.score ?? 0)); // แย่สุดก่อน

      const sellOrders = [];
      let sellProceeds = 0;

      for (const it of sellCandidates) {
        const h = holdingsBySymbol.get(it.stockSymbol);
        const qtyHeld = Math.floor(Number(h?.Quantity) || 0);
        if (qtyHeld <= 0) continue;

        // ถ้าต้องการขายบางส่วนเมื่อหลุด stop-loss ให้คำนวณ qtyToSell ตรงนี้
        const qtyToSell = qtyHeld; // ขายทั้งหมดตามดีฟอลต์

        const grossProceedUSD = qtyToSell * it.priceUSD;
        const feeUSD = grossProceedUSD * FEE_RATE;
        const netProceedUSD = toMoney2(grossProceedUSD - feeUSD);

        await connection.query(
          `INSERT INTO papertrade
             (PaperPortfolioID, TradeType, Quantity, Price, TradeDate, UserID, StockSymbol)
           VALUES (?, 'sell', ?, ?, NOW(), ?, ?)`,
          [
            portfolioId,
            qtyToSell,
            netProceedUSD,
            userId,
            normSymbol(it.stockSymbol),
          ]
        );

        const remain = (Number(h.Quantity) || 0) - qtyToSell;
        if (remain > 0) {
          await connection.query(
            `UPDATE paperportfolioholdings SET Quantity = ? WHERE PaperHoldingID = ?`,
            [remain, h.PaperHoldingID]
          );
        } else {
          await connection.query(
            `DELETE FROM paperportfolioholdings WHERE PaperHoldingID = ?`,
            [h.PaperHoldingID]
          );
        }

        sellProceeds = toMoney2(sellProceeds + netProceedUSD);

        sellOrders.push({
          stockSymbol: normSymbol(it.stockSymbol),
          quantity: qtyToSell,
          priceLocal: it.priceLocal,
          priceUSD: it.priceUSD,
          predictionLocal: it.predictionLocal,
          predictionUSD: it.predictionUSD,
          grossProceedUSD: toMoney2(grossProceedUSD),
          feeUSD: toMoney2(feeUSD),
          netProceedUSD,
          score: it.score,
          stockDetailId: it.stockDetailId,
          market: it.market,
          isThai: it.isThai,
          fxRate: it.fxRate,
        });
      }

      if (sellProceeds > 0) {
        await connection.query(
          `UPDATE papertradeportfolio
              SET autoBalance = autoBalance + ?
            WHERE PaperPortfolioID = ? AND UserID = ?`,
          [toMoney2(sellProceeds), portfolioId, userId]
        );
        autoBudget = toMoney2(autoBudget + sellProceeds);
      }

      // ===== BUY: score > 0, จัดงบตามน้ำหนัก score, ซื้อ = ราคา + ค่าธรรมเนียม =====
      const buyList = enriched
        .filter((x) => (x.score ?? 0) > 0 && Number.isFinite(x.priceUSD))
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

      const scoreSum = buyList.reduce(
        (s, x) => s + Math.max(0, x.score || 0),
        0
      );
      const equalWeight = buyList.length ? 1 / buyList.length : 0;

      const targetBudgets = buyList.map((x) => {
        const w =
          scoreSum > 0 ? Math.max(0, x.score || 0) / scoreSum : equalWeight;
        const budget = toMoney2(autoBudget * w);
        return { sym: normSymbol(x.stockSymbol), w, budget };
      });

      let remainingBudget = autoBudget;
      const buyOrders = [];
      let usedAmount = 0;

      // รอบแรก: ใช้งบตามน้ำหนัก
      for (let i = 0; i < buyList.length; i++) {
        const it = buyList[i];
        const tgt = targetBudgets[i];
        if (remainingBudget <= 0) break;

        const allowed = Math.min(tgt.budget, remainingBudget);
        const qty = Math.floor(allowed / it.priceUSD); // ซื้อด้วยราคา USD
        if (qty <= 0) continue;

        const grossCostUSD = qty * it.priceUSD;
        const feeUSD = grossCostUSD * FEE_RATE;
        const totalSpendUSD = toMoney2(grossCostUSD + feeUSD);

        await connection.query(
          `INSERT INTO papertrade
             (PaperPortfolioID, TradeType, Quantity, Price, TradeDate, UserID, StockSymbol)
           VALUES (?, 'buy', ?, ?, NOW(), ?, ?)`,
          [portfolioId, qty, totalSpendUSD, userId, normSymbol(it.stockSymbol)]
        );

        await upsertHolding(connection, {
          portfolioId,
          symbol: normSymbol(it.stockSymbol),
          buyQty: qty,
          priceUSD: it.priceUSD, // เก็บ BuyPrice เป็น USD/หุ้น
        });

        buyOrders.push({
          stockSymbol: normSymbol(it.stockSymbol),
          quantity: qty,
          priceLocal: it.priceLocal,
          priceUSD: it.priceUSD,
          predictionLocal: it.predictionLocal,
          predictionUSD: it.predictionUSD,
          grossCostUSD: toMoney2(grossCostUSD),
          feeUSD: toMoney2(feeUSD),
          totalSpendUSD,
          score: it.score,
          stockDetailId: it.stockDetailId,
          market: it.market,
          isThai: it.isThai,
          fxRate: it.fxRate,
        });

        remainingBudget = toMoney2(remainingBudget - totalSpendUSD);
        usedAmount = toMoney2(usedAmount + totalSpendUSD);
      }

      // รอบสอง: เก็บเศษงบ
      if (remainingBudget > 0) {
        for (const it of buyList) {
          if (remainingBudget <= 0) break;

          const qty = Math.floor(remainingBudget / it.priceUSD);
          if (qty <= 0) continue;

          const grossCostUSD = qty * it.priceUSD;
          const feeUSD = grossCostUSD * FEE_RATE;
          const totalSpendUSD = toMoney2(grossCostUSD + feeUSD);

          await connection.query(
            `INSERT INTO papertrade
               (PaperPortfolioID, TradeType, Quantity, Price, TradeDate, UserID, StockSymbol)
             VALUES (?, 'buy', ?, ?, NOW(), ?, ?)`,
            [
              portfolioId,
              qty,
              totalSpendUSD,
              userId,
              normSymbol(it.stockSymbol),
            ]
          );

          await upsertHolding(connection, {
            portfolioId,
            symbol: normSymbol(it.stockSymbol),
            buyQty: qty,
            priceUSD: it.priceUSD,
          });

          buyOrders.push({
            stockSymbol: normSymbol(it.stockSymbol),
            quantity: qty,
            priceLocal: it.priceLocal,
            priceUSD: it.priceUSD,
            predictionLocal: it.predictionLocal,
            predictionUSD: it.predictionUSD,
            grossCostUSD: toMoney2(grossCostUSD),
            feeUSD: toMoney2(feeUSD),
            totalSpendUSD,
            score: it.score,
            stockDetailId: it.stockDetailId,
            market: it.market,
            isThai: it.isThai,
            fxRate: it.fxRate,
          });

          remainingBudget = toMoney2(remainingBudget - totalSpendUSD);
          usedAmount = toMoney2(usedAmount + totalSpendUSD);
        }
      }

      // หัก autoBalance ด้วยเงินที่ใช้จริง (รวมค่าธรรมเนียม)
      if (usedAmount > 0) {
        const [upd] = await connection.query(
          `UPDATE papertradeportfolio
              SET autoBalance = GREATEST(0, autoBalance - ?)
            WHERE PaperPortfolioID = ? AND UserID = ?`,
          [toMoney2(usedAmount), portfolioId, userId]
        );
        if (!upd?.affectedRows) {
          await connection.rollback();
          results.push({
            portfolioId,
            userId,
            status: "FAILED",
            error: "ปรับปรุง autoBalance ไม่สำเร็จ",
          });
          continue;
        }
      }

      await connection.commit();

      results.push({
        status: "EXECUTED",
        userId,
        portfolio: {
          paperPortfolioId: portfolioId,
          balance,
          autoBalanceBefore: toMoney2(rawAutoBalance),
          buysCostUSD: toMoney2(usedAmount),
          autoBalanceAfter: toMoney2(
            (Number(rawAutoBalance) || 0) - usedAmount
          ),
          stopLossPct: stopLossPct,
          currency: PORTFOLIO_CCY,
        },
        counts: {
          consideredSymbols: enriched.length,
          sellPlaced: sellOrders.length,
          buyPlaced: buyOrders.length,
        },
        orders: { sells: sellOrders, buys: buyOrders },
      });
    } // end portfolios loop

    // ✅ Debug FX สำหรับหุ้นไทย
    const thaiConverted = enriched
      .filter((x) => x.isThai)
      .map((x) => ({
        stockSymbol: x.stockSymbol,
        fxRateTHB2USD: x.fxRate,
        priceTHB: x.priceLocal,
        priceUSD: x.priceUSD,
        predictionTHB: x.predictionLocal,
        predictionUSD: x.predictionUSD,
        score: x.score,
        stockDetailId: x.stockDetailId,
      }));

    return res.json({
      status: "DONE",
      note: "ใช้ stop-loss = leverage% ของพอร์ต เทียบจาก BuyPrice(USD). THB→USD เฉพาะหุ้นไทย. SELL = ราคา - fee เติม autoBalance. BUY = ราคา + fee หัก autoBalance. BuyPrice เก็บ USD/หุ้น.",
      thaiConverted,
      enrichedList: enriched.map((x) => ({
        stockSymbol: x.stockSymbol,
        market: x.market,
        isThai: x.isThai,
        fxRate: x.fxRate,
        priceLocal: x.priceLocal,
        priceUSD: x.priceUSD,
        predictionLocal: x.predictionLocal,
        predictionUSD: x.predictionUSD,
        score: x.score,
        stockDetailId: x.stockDetailId,
      })),
      portfolios: results,
    });
  } catch (err) {
    try {
      if (connection) await connection.rollback();
    } catch {}
    console.error("AutoTrade Error:", err);
    return res
      .status(500)
      .json({ error: "เกิดข้อผิดพลาด", detail: err?.message || String(err) });
  } finally {
    if (connection) connection.release();
  }
});

// POST /api/autoTrade
app.post("/api/autoTrade", verifyToken, async (req, res) => {
  let connection;

  const PORTFOLIO_CCY = "USD";
  const fxCache = new Map();

  async function getFxRate(from, to) {
    const key = `${from}->${to}`;
    const now = Date.now();
    const cached = fxCache.get(key);
    // TTL 10 นาที
    if (cached && now - cached.ts < 10 * 60 * 1000) return cached.rate;

    if (from === to) {
      fxCache.set(key, { rate: 1, ts: now });
      return 1;
    }
    try {
      const url = `https://api.exchangerate.host/convert?from=${encodeURIComponent(
        from
      )}&to=${encodeURIComponent(to)}`;
      const { data } = await axios.get(url, { timeout: 6000 });
      const rate = Number(data?.result);
      const finalRate = Number.isFinite(rate) && rate > 0 ? rate : 1;
      fxCache.set(key, { rate: finalRate, ts: now });
      return finalRate;
    } catch (e) {
      console.error("FX API error:", e?.message || e);
      return 1; // fallback ปลอดภัย
    }
  }

  // ตลาดไทย? → แปลง THB->USD; อื่น ๆ ใช้ USD ตรง ๆ
  function isThaiMarket(marketRaw = "") {
    return String(marketRaw).toLowerCase().includes("thailand");
  }
  function toTvMarket(marketRaw = "") {
    return isThaiMarket(marketRaw) ? "thailand" : "usa";
  }

  // ====== upsert holdings helper (ราคาเป็น USD) ======
  async function upsertHolding(
    conn,
    { portfolioId, symbol, buyQty, priceUSD }
  ) {
    const [rows] = await conn.query(
      `SELECT PaperHoldingID, Quantity, BuyPrice
         FROM paperportfolioholdings
        WHERE PaperPortfolioID = ? AND StockSymbol = ?
        FOR UPDATE`,
      [portfolioId, symbol]
    );

    if (rows && rows.length > 0) {
      const { PaperHoldingID, Quantity: oldQty, BuyPrice: oldPrice } = rows[0];
      const newQty = Number(oldQty) + Number(buyQty);
      const newBuyPrice =
        newQty > 0
          ? Number(
              (
                (Number(oldQty) * Number(oldPrice) +
                  Number(buyQty) * Number(priceUSD)) /
                newQty
              ).toFixed(6)
            )
          : Number(priceUSD.toFixed(6));
      await conn.query(
        `UPDATE paperportfolioholdings
            SET Quantity = ?, BuyPrice = ?, Type = 'AUTO'
          WHERE PaperHoldingID = ?`,
        [newQty, newBuyPrice, PaperHoldingID]
      );
      return {
        mode: "update",
        newQty,
        newBuyPrice,
        paperHoldingId: PaperHoldingID,
      };
    } else {
      const buyPrice = Number(priceUSD.toFixed(6));
      const [ins] = await conn.query(
        `INSERT INTO paperportfolioholdings
           (Quantity, BuyPrice, PaperPortfolioID, StockSymbol, Type)
         VALUES (?, ?, ?, ?, 'AUTO')`,
        [buyQty, buyPrice, portfolioId, symbol]
      );
      return {
        mode: "insert",
        newQty: buyQty,
        newBuyPrice: buyPrice,
        paperHoldingId: ins.insertId,
      };
    }
  }

  // mapLimit เล็ก ๆ ไว้รัน parallel แบบจำกัด concurrency
  const mapLimit = async (items, limit, worker) => {
    const out = new Array(items.length);
    let i = 0;
    const runners = Array(Math.min(limit, items.length))
      .fill(0)
      .map(async () => {
        while (true) {
          const idx = i++;
          if (idx >= items.length) break;
          out[idx] = await worker(items[idx], idx);
        }
      });
    await Promise.all(runners);
    return out;
  };

  try {
    const userId =
      req.userId ??
      req.user?.UserID ??
      req.user?.userId ??
      req.user?.id ??
      null;
    if (!userId)
      return res.status(401).json({ error: "ไม่พบ user id ใน token" });

    const DEFAULT_LIMIT = 50;
    const HARD_CAP_LIMIT = 200;
    const EFFECTIVE_LIMIT = Math.min(DEFAULT_LIMIT, HARD_CAP_LIMIT);
    const CONCURRENCY = 8;

    connection = await pool.promise().getConnection();

    // 1) portfolio (ต้อง ON)
    const [pfRows] = await connection.query(
      `SELECT PaperPortfolioID, Balance, autoBalance, status ,leverage
         FROM papertradeportfolio
        WHERE UserID = ?
        ORDER BY PaperPortfolioID DESC
        LIMIT 1`,
      [userId]
    );
    if (!pfRows || pfRows.length === 0) {
      return res
        .status(404)
        .json({ message: "ไม่พบ paper trade portfolio ของผู้ใช้" });
    }

    const {
      PaperPortfolioID: portfolioId,
      Balance: rawBalance,
      autoBalance: rawAutoBalance,
      status: rawStatus,
      leverage: rawAveragePct,
    } = pfRows[0];

    const isOn =
      String(rawStatus || "")
        .trim()
        .toLowerCase() === "on";
    if (!isOn) {
      return res.json({
        status: "SKIPPED",
        reason: "portfolio.status ไม่ได้เป็น ON",
        portfolio: {
          paperPortfolioId: portfolioId,
          balance: Number(rawBalance) || 0,
          autoBalance: Number(rawAutoBalance) || 0,
          status: rawStatus || null,
          leverage: rawLeverage || null,
          autoBalance: Number(rawAutoBalance) || 0,
          average: Number(rawAveragePct) || 0,
          status: rawStatus || null,
          currency: PORTFOLIO_CCY,
        },
      });
    }

    const balance = Number(rawBalance) || 0;
    const riskPct = Math.max(0, Number(rawAveragePct) || 0);
    let autoBudget = Math.max(0, Number(rawAutoBalance) || 0);
    if (autoBudget <= 0) {
      return res.json({
        status: "SKIPPED",
        reason: "autoBalance = 0",
        portfolio: {
          paperPortfolioId: portfolioId,
          balance,
          autoBalance: autoBudget,
          average: riskPct,
          status: rawStatus || null,
          currency: PORTFOLIO_CCY,
        },
      });
    }

    // NOTE: ถ้า autoBalance ใน DB เป็น THB ให้แปลงเป็น USD ก่อนใช้งาน:
    // const autoFx = await getFxRate("THB", "USD");
    // autoBudget = Number((autoBudget * autoFx).toFixed(2));

    // 2) last close + StockDetailID (ถ้าต้องการให้รันเฉพาะสัญลักษณ์ที่ส่งมา เติม WHERE s.StockSymbol = ? ได้)
    const [priceRows] = await connection.query(
      `
      SELECT
        s.StockSymbol,
        s.Market,
        sd.ClosePrice,
        sd.Date   AS DbPriceDate,
        sd.StockDetailID
      FROM stock s
      JOIN (
        SELECT sd1.StockSymbol, sd1.ClosePrice, sd1.Date, sd1.StockDetailID
        FROM stockdetail sd1
        JOIN (
          SELECT StockSymbol, MAX(Date) AS MaxDate
          FROM stockdetail
          GROUP BY StockSymbol
        ) m ON m.StockSymbol = sd1.StockSymbol AND m.MaxDate = sd1.Date
      ) sd ON sd.StockSymbol = s.StockSymbol
      ORDER BY s.StockSymbol
      LIMIT ?`,
      [EFFECTIVE_LIMIT]
    );
    if (!priceRows || priceRows.length === 0) {
      return res
        .status(404)
        .json({ message: "ไม่พบข้อมูลหุ้นหรือราคาปิดล่าสุด" });
    }

    // 3) fetch realtime price & decide action (แปลง THB->USD เฉพาะหุ้นไทย)
    const enriched = await mapLimit(priceRows, CONCURRENCY, async (row) => {
      const symbol = String(row.StockSymbol || "")
        .replace(/\.BK$/i, "")
        .toUpperCase();
      const marketRaw = row.Market || "";
      const tvMarket = toTvMarket(marketRaw);

      // 3.1 ราคา realtime (native currency)
      let apiPrice = null;
      let apiError = null;
      try {
        const r = await getTradingViewPrice(symbol, tvMarket); // ต้องมีฟังก์ชันนี้อยู่แล้ว
        const p = Number(r?.price);
        if (Number.isFinite(p) && p > 0) apiPrice = p;
        else apiError = "invalid_price";
      } catch (e) {
        apiError = e?.message || "api_error";
      }

      // 3.2 แปลงเฉพาะหุ้นไทย -> USD
      const dbClose = Number(row.ClosePrice);
      const isThai = isThaiMarket(marketRaw);
      const fxRate = isThai ? await getFxRate("THB", "USD") : 1;

      const priceUSD =
        apiPrice != null && fxRate
          ? Number((apiPrice * fxRate).toFixed(6))
          : null;

      const dbCloseUSD =
        Number.isFinite(dbClose) && fxRate
          ? Number((dbClose * fxRate).toFixed(6))
          : null;

      // 3.3 สร้างสัญญาณจากราคา USD (หลังแปลงแล้ว)
      let percentDiff = null;
      let action = null;
      if (priceUSD != null && dbCloseUSD != null && dbCloseUSD > 0) {
        percentDiff = ((priceUSD - dbCloseUSD) / dbCloseUSD) * 100;
        action = percentDiff > 5 ? "BUY" : "SELL";
      }

      return {
        stockSymbol: symbol,
        market: marketRaw,
        stockDetailId: row.StockDetailID,
        apiError,
        isThai,
        fxRate, // THB->USD หรือ 1
        priceUSD, // ใช้ตัวนี้ต่อทั้งหมด
        dbCloseUSD,
        percentDiff:
          percentDiff == null ? null : Number(percentDiff.toFixed(2)),
        action,
      };
    });

    // 4) BUY only + weight by percentDiff
    const buyList = enriched
      .filter((x) => x && x.action === "BUY" && Number.isFinite(x.priceUSD))
      .sort((a, b) => (b.percentDiff ?? -1) - (a.percentDiff ?? -1));

    if (buyList.length === 0) {
      return res.json({
        status: "EXECUTED",
        note: "ไม่มีสัญญาณ BUY หรือราคาปัจจุบันไม่พร้อม",
        portfolio: {
          paperPortfolioId: portfolioId,
          balance,
          autoBalanceBefore: autoBudget,
          autoBalanceUsed: 0,
          autoBalanceAfter: autoBudget,
          average: riskPct,
          currency: PORTFOLIO_CCY,
        },
        counts: {
          considered: enriched.length,
          placed: 0,
          skipped: enriched.length,
        },
        orders: [],
      });
    }

    let weightSum = buyList.reduce(
      (s, x) => s + Math.max(0, x.percentDiff || 0),
      0
    );
    const equalWeight = 1 / buyList.length;
    const targetBudgets = buyList.map((x) => {
      const w =
        weightSum > 0
          ? Math.max(0, x.percentDiff || 0) / weightSum
          : equalWeight;
      const budget = Number((autoBudget * w).toFixed(2)); // USD
      return { sym: x.stockSymbol, w, budget };
    });

    await connection.beginTransaction();

    let remainingBudget = autoBudget;
    const orders = [];
    let usedAmount = 0;

    // รอบแรก: ใช้งบตามน้ำหนัก (ราคาที่ใช้ = USD)
    for (let i = 0; i < buyList.length; i++) {
      const it = buyList[i];
      const tgt = targetBudgets[i];
      if (remainingBudget <= 0) break;

      const allowed = Math.min(tgt.budget, remainingBudget);
      const qty = Math.floor(allowed / it.priceUSD);
      if (qty <= 0) continue;

      const costUSD = Number((qty * it.priceUSD).toFixed(5));

      await connection.query(
        `INSERT INTO autotrade
           (PaperPortfolioID, TradeType, Quantity, Price, StockDetailID, Status, TradeDate, UserID, StockSymbol)
         VALUES (?, 'BUY', ?, ?, ?, 'BUY', NOW(), ?, ?)`,
        [
          portfolioId,
          qty,
          it.priceUSD,
          it.stockDetailId,
          userId,
          it.stockSymbol,
        ]
      );

      await upsertHolding(connection, {
        portfolioId,
        symbol: it.stockSymbol,
        buyQty: qty,
        priceUSD: it.priceUSD,
      });

      orders.push({
        stockSymbol: it.stockSymbol,
        priceUSD: it.priceUSD,
        quantity: qty,
        costUSD,
        percentDiff: it.percentDiff,
        stockDetailId: it.stockDetailId,
        isThai: it.isThai,
        fxRate: it.fxRate,
      });

      remainingBudget = Number((remainingBudget - costUSD).toFixed(2));
      usedAmount = Number((usedAmount + costUSD).toFixed(2));
    }

    // รอบสอง: ใช้งบเศษที่เหลือ ไล่ซื้อตามลำดับ
    if (remainingBudget > 0) {
      for (const it of buyList) {
        if (remainingBudget <= 0) break;
        const qty = Math.floor(remainingBudget / it.priceUSD);
        if (qty <= 0) continue;

        const costUSD = Number((qty * it.priceUSD).toFixed(5));

        await connection.query(
          `INSERT INTO autotrade
             (PaperPortfolioID, TradeType, Quantity, Price, StockDetailID, Status, TradeDate, UserID, StockSymbol)
           VALUES (?, 'BUY', ?, ?, ?, 'BUY', NOW(), ?, ?)`,
          [
            portfolioId,
            qty,
            it.priceUSD,
            it.stockDetailId,
            userId,
            it.stockSymbol,
          ]
        );

        await upsertHolding(connection, {
          portfolioId,
          symbol: it.stockSymbol,
          buyQty: qty,
          priceUSD: it.priceUSD,
        });

        orders.push({
          stockSymbol: it.stockSymbol,
          priceUSD: it.priceUSD,
          quantity: qty,
          costUSD,
          percentDiff: it.percentDiff,
          stockDetailId: it.stockDetailId,
          isThai: it.isThai,
          fxRate: it.fxRate,
        });

        remainingBudget = Number((remainingBudget - costUSD).toFixed(2));
        usedAmount = Number((usedAmount + costUSD).toFixed(2));
      }
    }

    if (usedAmount > 0) {
      const [upd] = await connection.query(
        `UPDATE papertradeportfolio
            SET autoBalance = GREATEST(0, autoBalance - ?)
          WHERE PaperPortfolioID = ? AND UserID = ?`,
        [usedAmount, portfolioId, userId]
      );
      if (!upd || !upd.affectedRows) {
        await connection.rollback();
        return res
          .status(409)
          .json({ error: "ปรับปรุง autoBalance ไม่สำเร็จ (race condition)" });
      }
    }

    await connection.commit();

    return res.json({
      status: "EXECUTED",
      userId,
      portfolio: {
        paperPortfolioId: portfolioId,
        balance,
        autoBalanceBefore: autoBudget,
        autoBalanceUsed: usedAmount,
        autoBalanceAfter: Number((autoBudget - usedAmount).toFixed(2)),
        average: riskPct,
        currency: PORTFOLIO_CCY, // USD
      },
      counts: {
        considered: enriched.length,
        buyCandidates: buyList.length,
        placed: orders.length,
        skipped: enriched.length - orders.length,
      },
      allocationPreview: targetBudgets,
      orders,
      note: "หุ้นไทยถูกแปลง THB→USD ก่อนคำนวณ/บันทึก; เก็บราคาใน DB เป็น USD",
    });
  } catch (err) {
    try {
      if (connection) await connection.rollback();
    } catch {}
    console.error(err);
    return res
      .status(500)
      .json({ error: "เกิดข้อผิดพลาด", detail: err?.message || String(err) });
  } finally {
    if (connection) connection.release();
  }
});

app.patch("/api/autotrade-setting", verifyToken, async (req, res) => {
  let connection;
  try {
    let { balance, autoBalance, status, average } = req.body;

    const toNum = (v) =>
      v === null || v === undefined || v === "" ? undefined : Number(v);
    const round2 = (n) => Math.round(n * 100) / 100;

    balance = toNum(balance);
    autoBalance = toNum(autoBalance);
    average = toNum(average);

    // --- Validation ---
    if (balance !== undefined && !Number.isFinite(balance)) {
      return res.status(400).json({ error: "balance must be a number" });
    }
    if (autoBalance !== undefined && !Number.isFinite(autoBalance)) {
      return res.status(400).json({ error: "autoBalance must be a number" });
    }
    if (average !== undefined && !Number.isInteger(average)) {
      return res.status(400).json({ error: "average must be an integer" });
    }
    if (status !== undefined) {
      const normalized = String(status).toUpperCase();
      if (!["ON", "OFF"].includes(normalized)) {
        return res.status(400).json({ error: "status must be 'ON' or 'OFF'" });
      }
      status = normalized;
    }
    if ([balance, autoBalance, average, status].every((v) => v === undefined)) {
      return res.status(400).json({
        error:
          "Please provide at least one field to update: balance, autoBalance, average, or status",
      });
    }

    connection = await pool.promise().getConnection();
    await connection.beginTransaction();

    // --- Find portfolio ---
    const [rows] = await connection.query(
      `SELECT PaperPortfolioID
         FROM papertradeportfolio
        WHERE UserID = ? LIMIT 1`,
      [req.userId]
    );
    if (!rows || rows.length === 0) {
      await connection.rollback();
      return res.status(404).json({ error: "User portfolio not found" });
    }
    const portfolioId = rows[0].PaperPortfolioID;

    // --- Lock + read current ---
    const [currArr] = await connection.query(
      `SELECT Balance AS currBalance,
              autoBalance AS currAuto,
              Status    AS currStatus,
              leverage  AS currLeverage
         FROM papertradeportfolio
        WHERE PaperPortfolioID = ? AND UserID = ?
        FOR UPDATE`,
      [portfolioId, req.userId]
    );
    if (!currArr || currArr.length === 0) {
      await connection.rollback();
      return res.status(404).json({ error: "Portfolio not found" });
    }

    let currBalance = Number(currArr[0].currBalance) || 0;
    let currAuto = Number(currArr[0].currAuto) || 0;

    // --- Compute updates ---
    let newBalance = currBalance;
    let newAuto = currAuto;
    let touchedMoney = false;

    if (autoBalance !== undefined) {
      // TRANSFER MODE: target auto = requested autoBalance
      const targetAuto = round2(autoBalance);
      if (targetAuto < 0) {
        await connection.rollback();
        return res
          .status(400)
          .json({ error: "autoBalance cannot be negative" });
      }

      const delta = round2(targetAuto - currAuto); // + = move Demo -> Auto, - = Auto -> Demo

      if (delta > 0) {
        // Need to move delta from Balance to Auto
        if (round2(currBalance - delta) < 0) {
          await connection.rollback();
          return res.status(400).json({
            error: "Insufficient balance to increase autoBalance",
            detail: { currBalance, currAuto, targetAuto, required: delta },
          });
        }
        newBalance = round2(currBalance - delta);
        newAuto = round2(currAuto + delta); // == targetAuto
      } else if (delta < 0) {
        // Move back |delta| from Auto to Balance
        const abs = Math.abs(delta);
        if (round2(currAuto - abs) < 0) {
          await connection.rollback();
          return res.status(400).json({
            error: "autoBalance would become negative",
            detail: { currBalance, currAuto, targetAuto },
          });
        }
        newBalance = round2(currBalance + abs);
        newAuto = round2(currAuto - abs); // == targetAuto
      } else {
        // target equals current -> no money movement
        newBalance = currBalance;
        newAuto = currAuto;
      }

      touchedMoney = true;

      // If client also sent balance simultaneously, we ignore it to keep transfer semantics consistent.
    } else if (balance !== undefined) {
      // DIRECT BALANCE UPDATE (no auto movement)
      const targetBal = round2(balance);
      if (targetBal < 0) {
        await connection.rollback();
        return res.status(400).json({ error: "balance cannot be negative" });
      }
      newBalance = targetBal;
      newAuto = currAuto;
      touchedMoney = true;
    }

    // --- Build update query ---
    const setParts = [];
    const params = [];

    if (touchedMoney) {
      setParts.push("Balance = ?");
      params.push(newBalance);
      setParts.push("autoBalance = ?");
      params.push(newAuto);
    }

    if (average !== undefined) {
      setParts.push("leverage = ?"); // map 'average' -> 'leverage'
      params.push(average);
    }
    if (status !== undefined) {
      setParts.push("Status = ?");
      params.push(status);
    }

    if (setParts.length === 0) {
      await connection.rollback();
      return res.status(400).json({ error: "No valid fields to update" });
    }

    params.push(portfolioId, req.userId);

    await connection.query(
      `UPDATE papertradeportfolio
          SET ${setParts.join(", ")}
        WHERE PaperPortfolioID = ? AND UserID = ?`,
      params
    );

    await connection.commit();

    // --- Return updated row ---
    const [after] = await connection.query(
      `SELECT PaperPortfolioID, UserID, Balance, autoBalance, Status, leverage
         FROM papertradeportfolio
        WHERE PaperPortfolioID = ? AND UserID = ?`,
      [portfolioId, req.userId]
    );

    return res.json(after[0]);
  } catch (error) {
    if (connection) await connection.rollback();
    console.error("PATCH /api/autotrade-setting error:", error);
    return res.status(500).json({ error: "Server error" });
  } finally {
    if (connection) connection.release();
  }
});

app.post("/api/portfolio/trade", verifyToken, async (req, res) => {
  let connection;
  try {
    let { stockSymbol, quantity, tradeType } = req.body;
    const userId = req.userId;
    if (
      !stockSymbol ||
      !quantity ||
      !tradeType ||
      !["buy", "sell"].includes(tradeType)
    ) {
      return res.status(400).json({
        error:
          "กรุณาระบุข้อมูลให้ครบถ้วน: stockSymbol, quantity, และ tradeType ('buy' หรือ 'sell')",
      });
    }
    const parsedQuantity = parseInt(quantity, 10);
    if (isNaN(parsedQuantity) || parsedQuantity <= 0) {
      return res.status(400).json({ error: "จำนวนหุ้นไม่ถูกต้อง" });
    }
    const normalizedSymbol = stockSymbol.toUpperCase().replace(".BK", "");

    // === CONFIG ===
    // ✅ ซื้อไม่คิดค่าธรรมเนียม
    const FEE_BUY_RATE = 0.0;
    // ✅ ค่าธรรมเนียมเฉพาะตอนขาย (ตัวอย่าง 0.15%)
    const FEE_SELL_RATE = 0.0015;

    // เริ่ม Transaction
    connection = await pool.promise().getConnection();
    await connection.beginTransaction();

    // 2) ตรวจตลาด
    const [stockInfoRows] = await connection.query(
      "SELECT Market FROM Stock WHERE StockSymbol = ?",
      [normalizedSymbol]
    );
    if (stockInfoRows.length === 0) {
      await connection.rollback();
      return res
        .status(404)
        .json({ error: `ไม่พบข้อมูลหุ้น ${normalizedSymbol}` });
    }
    const market = stockInfoRows[0].Market;

    // 3) ราคาปัจจุบัน
    let currentPrice;
    try {
      const tradingViewMarket = market === "Thailand" ? "thailand" : "usa";
      const priceData = await getTradingViewPrice(
        normalizedSymbol,
        tradingViewMarket
      );
      currentPrice = Number(priceData.price);
      if (!Number.isFinite(currentPrice) || currentPrice <= 0)
        throw new Error("bad price");
    } catch (e) {
      await connection.rollback();
      console.error("TradingView API error:", e.message);
      return res
        .status(500)
        .json({ error: `เกิดข้อผิดพลาดในการดึงราคาหุ้น ${normalizedSymbol}` });
    }

    // 4) พอร์ต + FX
    const [portfolioRows] = await connection.query(
      "SELECT * FROM papertradeportfolio WHERE UserID = ?",
      [userId]
    );
    if (portfolioRows.length === 0) {
      await connection.rollback();
      return res
        .status(404)
        .json({ message: "ไม่พบพอร์ตการลงทุน กรุณาสร้างพอร์ตก่อน" });
    }
    const portfolio = portfolioRows[0];
    const portfolioId = portfolio.PaperPortfolioID;
    let balanceUSD = Number(portfolio.Balance) || 0;

    const isThaiStock = market === "Thailand";
    let thbToUsdRate = 1;
    if (isThaiStock) {
      thbToUsdRate = await getThbToUsdRate(); // เช่น 0.027xx (USD/THB)
    }

    // ราคา USD ต่อหน่วย ที่จะเก็บใน holdings (ไทยต้องคูณ FX)
    const unitPriceUSD = isThaiStock
      ? currentPrice * thbToUsdRate
      : currentPrice;

    // มูลค่ารวมก่อนค่าธรรมเนียม (gross)
    const grossUSD = unitPriceUSD * parsedQuantity;

    // === คิดค่าธรรมเนียมตามขา ===
    const feeUSD =
      tradeType === "sell" ? grossUSD * FEE_SELL_RATE : grossUSD * FEE_BUY_RATE; // (BUY = 0)

    // === NET ที่จะใช้ “ตัด/เพิ่มเงิน” และ “save ลง papertrade.Price” ===
    const netAmountUSD = tradeType === "buy" ? grossUSD : grossUSD - feeUSD; // ✅ BUY ไม่หัก fee, SELL หัก fee

    if (tradeType === "buy") {
      // --- BUY ---
      if (balanceUSD < netAmountUSD) {
        await connection.rollback();
        return res.status(400).json({ error: "ยอดเงินคงเหลือไม่เพียงพอ" });
      }

      // ✅ หัก "สุทธิ" (ซึ่ง = gross เพราะ fee = 0)
      balanceUSD = Number((balanceUSD - netAmountUSD).toFixed(2));
      await connection.query(
        "UPDATE papertradeportfolio SET Balance = ? WHERE PaperPortfolioID = ?",
        [balanceUSD, portfolioId]
      );

      // เก็บล็อตเป็น USD/หน่วย เพื่อให้ตรงกับ Balance
      await connection.query(
        "INSERT INTO paperportfolioholdings (PaperPortfolioID, StockSymbol, Quantity, BuyPrice) VALUES (?, ?, ?, ?)",
        [portfolioId, normalizedSymbol, parsedQuantity, unitPriceUSD]
      );
    } else {
      // --- SELL ---
      const [holdingRows] = await connection.query(
        "SELECT * FROM paperportfolioholdings WHERE PaperPortfolioID = ? AND StockSymbol = ? ORDER BY PaperHoldingID ASC",
        [portfolioId, normalizedSymbol]
      );
      const totalHeldQuantity = holdingRows.reduce(
        (sum, row) => sum + Number(row.Quantity),
        0
      );
      if (totalHeldQuantity < parsedQuantity) {
        await connection.rollback();
        return res
          .status(400)
          .json({ error: "จำนวนหุ้นที่ต้องการขายไม่เพียงพอ" });
      }

      // ✅ บวก "สุทธิ" (gross - fee) เข้าบัญชี
      balanceUSD = Number((balanceUSD + netAmountUSD).toFixed(2));
      await connection.query(
        "UPDATE papertradeportfolio SET Balance = ? WHERE PaperPortfolioID = ?",
        [balanceUSD, portfolioId]
      );

      // ตัด FIFO lots
      let qty = parsedQuantity;
      for (const holding of holdingRows) {
        if (qty <= 0) break;
        const take = Math.min(qty, Number(holding.Quantity));
        qty -= take;
        const remain = Number(holding.Quantity) - take;
        if (remain > 0) {
          await connection.query(
            "UPDATE paperportfolioholdings SET Quantity = ? WHERE PaperHoldingID = ?",
            [remain, holding.PaperHoldingID]
          );
        } else {
          await connection.query(
            "DELETE FROM paperportfolioholdings WHERE PaperHoldingID = ?",
            [holding.PaperHoldingID]
          );
        }
      }
    }

    // 5) บันทึกประวัติ trade — เซฟ “ยอดสุทธิรวม” ลง Price ให้ตรงกับ Balance ที่ขยับจริง
    await connection.query(
      "INSERT INTO papertrade (PaperPortfolioID, StockSymbol, TradeType, Quantity, Price, TradeDate, UserID) VALUES (?, ?, ?, ?, ?, NOW(), ?)",
      [
        portfolioId,
        normalizedSymbol,
        tradeType,
        parsedQuantity,
        Number(netAmountUSD.toFixed(2)),
        userId,
      ]
    );

    await connection.commit();
    return res.status(200).json({
      message: `ทำรายการ ${tradeType === "buy" ? "ซื้อ" : "ขาย"} สำเร็จ`,
      trade: {
        type: tradeType,
        symbol: normalizedSymbol,
        market,
        marketPrice: Number(currentPrice), // THB ถ้าไทย / USD ถ้า US
        marketPriceCurrency: isThaiStock ? "THB" : "USD",
        unitPriceUSD: Number(unitPriceUSD.toFixed(6)), // ราคา USD/หน่วย ที่เก็บใน holdings
        grossUSD: Number(grossUSD.toFixed(2)), // มูลค่าก่อน fee
        feeUSD: Number(feeUSD.toFixed(2)), // BUY = 0, SELL = gross*rate
        netUSD: Number(netAmountUSD.toFixed(2)), // มูลค่าที่ใช้ตัด/เพิ่มเงินและเซฟลง papertrade
      },
      balanceUSD,
    });
  } catch (error) {
    if (connection) await connection.rollback();
    console.error("Error executing trade:", error);
    return res.status(500).json({ error: "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์" });
  } finally {
    if (connection) connection.release();
  }
});

app.get("/api/portfolio/history", verifyToken, async (req, res) => {
  let connection;
  try {
    connection = await pool.promise().getConnection();
    const userId = req.userId;

    const [portfolioRows] = await connection.query(
      "SELECT PaperPortfolioID FROM papertradeportfolio WHERE UserID = ?",
      [userId]
    );

    if (portfolioRows.length === 0) {
      return res.status(200).json({
        message: "ไม่พบพอร์ตการลงทุน จึงไม่มีประวัติการทำรายการ",
        data: [],
      });
    }

    const portfolioId = portfolioRows[0].PaperPortfolioID;

    const [transactions] = await connection.query(
      `SELECT StockSymbol, TradeType, Quantity, Price, TradeDate
       FROM papertrade
       WHERE UserID = ?
       ORDER BY TradeDate DESC`,
      [userId]
    );

    res.status(200).json({
      message: "ดึงข้อมูลประวัติการทำรายการสำเร็จ",
      data: transactions,
    });
  } catch (error) {
    console.error("Error fetching transaction history:", error);
    res.status(500).json({ error: "Internal server error" });
  } finally {
    if (connection) connection.release();
  }
});

app.post("/api/create-demo", verifyToken, async (req, res) => {
  let connection;
  try {
    // Use the promise-wrapped pool for async/await
    connection = await pool.promise().getConnection();
    const { amount } = req.body;

    // Check for required data, allowing amount to be 0
    if (amount === undefined) {
      return res.status(400).json({ error: "Amount is required" });
    }

    // Validate that amount is a valid number
    const parsedAmount = parseFloat(amount);
    if (isNaN(parsedAmount) || parsedAmount < 0) {
      return res.status(400).json({ error: "Invalid amount" });
    }

    // Check if the user already has a portfolio
    const [existingRows] = await connection.query(
      "SELECT UserID FROM papertradeportfolio WHERE UserID = ?",
      [req.userId]
    );

    if (existingRows.length > 0) {
      return res.status(400).json({ message: "User already has a portfolio" });
    }

    // Create a new portfolio with the specified amount
    await connection.query(
      "INSERT INTO papertradeportfolio (UserID, Balance) VALUES (?, ?)",
      [req.userId, parsedAmount]
    );

    res.status(201).json({
      message: "Portfolio created successfully",
      data: {
        UserID: req.userId,
        Balance: parsedAmount,
        createdAt: new Date(),
      },
    });
  } catch (error) {
    console.error("Error creating portfolio:", error);
    res.status(500).json({ error: "Internal server error" });
  } finally {
    if (connection) connection.release();
  }
});

app.put("/api/edit-demo", verifyToken, async (req, res) => {
  let connection;
  try {
    connection = await pool.promise().getConnection();

    const userId =
      req.user?.UserID ?? req.user?.userId ?? req.user?.id ?? req.userId;
    if (!userId) {
      return res.status(401).json({ error: "Unauthorized: missing user id" });
    }

    // รับ amount (รับได้ทั้ง string/number)
    const { amount } = req.body;
    if (amount === undefined || amount === null || amount === "") {
      return res.status(400).json({ error: "Amount is required" });
    }

    const n = Number(amount);
    if (!Number.isFinite(n) || n < 0) {
      return res.status(400).json({ error: "Invalid amount" });
    }

    // บังคับทศนิยม 2 ตำแหน่งด้วยหน่วยย่อย (เซ็นต์/สตางค์)
    const cents = Math.round(n * 100);
    const parsedAmount = cents / 100;

    // เริ่ม transaction
    await connection.beginTransaction();

    // ล็อกแถว portfolio ของผู้ใช้ (กัน race)
    const [rows] = await connection.query(
      "SELECT PaperPortfolioID, Balance FROM papertradeportfolio WHERE UserID = ? FOR UPDATE",
      [userId]
    );
    if (!rows || rows.length === 0) {
      await connection.rollback();
      return res.status(404).json({ error: "Portfolio not found" });
    }

    // อัปเดตยอด (ปัดแล้ว 2 ตำแหน่ง)
    await connection.query(
      "UPDATE papertradeportfolio SET Balance = ? WHERE UserID = ?",
      [parsedAmount, userId]
    );

    await connection.commit();

    return res.json({
      message: "Balance updated successfully",
      data: {
        UserID: userId,
        Balance: parsedAmount,
        updatedAt: new Date(), // ถ้าต้องการเวลา DB ใช้ SELECT NOW() เพิ่มเติมได้
      },
    });
  } catch (error) {
    console.error("Error updating balance:", error);
    if (connection) {
      try {
        await connection.rollback();
      } catch (_) {}
    }
    return res.status(500).json({ error: "Internal server error" });
  } finally {
    if (connection) connection.release();
  }
});

//-----------------------------------------------------------------------------------------------------------------------------------------------//

const verifyAdmin = (req, res, next) => {
  if (req.role !== "admin") {
    return res.status(403).json({ error: "Unauthorized access" });
  }
  next();
};

//Admin//
app.post("/api/admin/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "กรุณากรอกอีเมลและรหัสผ่าน" });
    }

    // ค้นหาผู้ใช้ที่เป็น Admin และมีสถานะ Active
    const sql =
      "SELECT * FROM User WHERE Email = ? AND Status = 'active' AND Role = 'admin'";
    pool.query(sql, [email], (err, results) => {
      if (err) {
        console.error("Database error during admin login:", err);
        return res
          .status(500)
          .json({ error: "เกิดข้อผิดพลาดระหว่างการเข้าสู่ระบบ" });
      }

      if (results.length === 0) {
        return res
          .status(404)
          .json({ message: "ไม่พบบัญชีแอดมิน หรืออาจถูกระงับ" });
      }

      const user = results[0];

      // ตรวจสอบรหัสผ่าน
      bcrypt.compare(password, user.Password, (err, isMatch) => {
        if (err) {
          console.error("Password comparison error:", err);
          return res
            .status(500)
            .json({ error: "เกิดข้อผิดพลาดในการตรวจสอบรหัสผ่าน" });
        }

        if (!isMatch) {
          return res
            .status(401)
            .json({ message: "อีเมลหรือรหัสผ่านไม่ถูกต้อง" });
        }

        // ✅ สร้าง JWT Token (ไม่มี LastLogin / LastLoginIP)
        const token = jwt.sign(
          { id: user.UserID, role: user.Role },
          JWT_SECRET,
          { expiresIn: "7d" }
        );

        // ✅ ส่งข้อมูล Response
        res.status(200).json({
          message: "เข้าสู่ระบบแอดมินสำเร็จ",
          token,
          user: {
            id: user.UserID,
            email: user.Email,
            username: user.Username,
            profile_image: user.ProfileImageURL,
            role: user.Role,
          },
        });
      });
    });
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//=====================================================================================================//
// 												ADMIN - USER MANAGEMENT API
//=====================================================================================================//

//=====================================================================================================//
//  ADMIN - USER MANAGEMENT API
//=====================================================================================================//

// ดึงรายชื่อผู้ใช้ทั้งหมด (พร้อมแบ่งหน้า + ค้นหา)
app.get("/api/admin/users", verifyToken, verifyAdmin, async (req, res) => {
  try {
    const page = Math.max(parseInt(req.query.page) || 1, 1);
    const limit = Math.min(Math.max(parseInt(req.query.limit) || 10, 1), 100);
    const offset = (page - 1) * limit;
    const searchTerm = req.query.search || "";

    const whereClause = searchTerm
      ? `WHERE (Username LIKE ? OR Email LIKE ?)`
      : "";
    const searchParams = searchTerm
      ? [`%${searchTerm}%`, `%${searchTerm}%`]
      : [];

    const countSql = `SELECT COUNT(*) as total FROM \`User\` ${whereClause}`;
    const dataSql = `
            SELECT UserID, Username, Email, Role, Status
            FROM \`User\`
            ${whereClause}
            ORDER BY UserID DESC
            LIMIT ? OFFSET ?
        `;

    const [countResult] = await pool.promise().query(countSql, searchParams);
    const [usersResult] = await pool
      .promise()
      .query(dataSql, [...searchParams, limit, offset]);

    const totalUsers = countResult[0].total;
    const totalPages = Math.ceil(totalUsers / limit);

    res.status(200).json({
      message: "Successfully retrieved users",
      data: usersResult,
      pagination: {
        currentPage: page,
        totalPages,
        totalUsers,
        limit,
      },
    });
  } catch (err) {
    console.error("Database error fetching users:", err);
    res.status(500).json({ error: "Database error while fetching users" });
  }
});

// อัปเดตสถานะผู้ใช้ (active/suspended)
app.put(
  "/api/admin/users/:userId/status",
  verifyToken,
  verifyAdmin,
  async (req, res) => {
    try {
      const { userId } = req.params;
      const { status } = req.body;

      if (!status || !["active", "suspended"].includes(status.toLowerCase())) {
        return res
          .status(400)
          .json({ error: "Invalid status. Must be 'active' or 'suspended'." });
      }

      if (parseInt(userId, 10) === req.userId) {
        return res
          .status(403)
          .json({ error: "Admins cannot change their own status." });
      }

      const sql = "UPDATE `User` SET Status = ? WHERE UserID = ?";
      const [result] = await pool
        .promise()
        .query(sql, [status.toLowerCase(), userId]);

      if (result.affectedRows === 0) {
        return res.status(404).json({ error: "User not found" });
      }

      res
        .status(200)
        .json({ message: `User status successfully updated to ${status}` });
    } catch (err) {
      console.error("Database error updating user status:", err);
      res.status(500).json({ error: "Database error" });
    }
  }
);

// แก้ไขข้อมูลผู้ใช้ (Username, Email, Role)
app.put(
  "/api/admin/users/:userId",
  verifyToken,
  verifyAdmin,
  async (req, res) => {
    try {
      const { userId } = req.params;
      const { username, email, role } = req.body;

      if (!username && !email && !role) {
        return res.status(400).json({ error: "No data provided for update." });
      }

      let updateFields = [];
      let params = [];

      if (username) {
        updateFields.push("Username = ?");
        params.push(username);
      }
      if (email) {
        updateFields.push("Email = ?");
        params.push(email);
      }
      if (role) {
        updateFields.push("Role = ?");
        params.push(role);
      }

      params.push(userId);

      const sql = `UPDATE \`User\` SET ${updateFields.join(
        ", "
      )} WHERE UserID = ?`;
      const [result] = await pool.promise().query(sql, params);

      if (result.affectedRows === 0) {
        return res.status(404).json({ error: "User not found" });
      }

      // คืนข้อมูล user หลังแก้ไข
      const [rows] = await pool
        .promise()
        .query(
          `SELECT UserID, Username, Email, Role, Status FROM \`User\` WHERE UserID = ?`,
          [userId]
        );

      res
        .status(200)
        .json({ message: "User profile updated successfully", data: rows[0] });
    } catch (err) {
      if (err.code === "ER_DUP_ENTRY") {
        const message = err.message.includes("Username")
          ? "This username is already in use."
          : "This email is already in use.";
        return res.status(409).json({ error: message });
      }
      console.error("Database error updating user:", err);
      return res
        .status(500)
        .json({ error: "Database error while updating user." });
    }
  }
);

// ลบผู้ใช้
app.delete(
  "/api/admin/users/:userId",
  verifyToken,
  verifyAdmin,
  async (req, res) => {
    try {
      const { userId } = req.params;

      if (parseInt(userId, 10) === req.userId) {
        return res
          .status(403)
          .json({ error: "Admins cannot delete their own account." });
      }

      const sql = "DELETE FROM `User` WHERE UserID = ?";
      const [result] = await pool.promise().query(sql, [userId]);

      if (result.affectedRows === 0) {
        return res.status(404).json({ error: "User not found" });
      }

      res.status(200).json({ message: "User deleted successfully" });
    } catch (err) {
      console.error("Database error deleting user:", err);
      if (err.code === "ER_ROW_IS_REFERENCED_2") {
        return res.status(400).json({
          error: "Cannot delete user. The user is linked to other data.",
        });
      }
      return res
        .status(500)
        .json({ error: "Database error while deleting user." });
    }
  }
);

//=====================================================================================================//
//  ADMIN - SIMPLE USER HOLDINGS (ดูหุ้นที่ผู้ใช้ถืออยู่แบบตรง ๆ)
//  คืน: StockSymbol, Quantity, BuyPrice, PaperPortfolioID (ตามจริงจาก holdings)
//  ตัวเลือก: ?symbol=AMD (กรองสัญลักษณ์), ?page=1&limit=50 (ถ้าข้อมูลเยอะ)
//=====================================================================================================//
app.get(
  "/api/admin/users/:userId/holdings-simple",
  verifyToken,
  verifyAdmin,
  async (req, res) => {
    const db = pool.promise();
    try {
      const userId = parseInt(req.params.userId, 10);
      if (!Number.isInteger(userId))
        return res.status(400).json({ error: "Invalid userId" });

      // optional
      const symbol = req.query.symbol?.trim();
      const page = Math.max(parseInt(req.query.page) || 1, 1);
      const limit = Math.min(
        Math.max(parseInt(req.query.limit) || 100, 1),
        500
      );
      const offset = (page - 1) * limit;

      // base where
      const where = ["ptp.UserID = ?"];
      const params = [userId];

      if (symbol) {
        where.push("pph.StockSymbol = ?");
        params.push(symbol);
      }

      // นับจำนวน (รองรับ pagination)
      const countSql = `
      SELECT COUNT(*) AS total
      FROM paperportfolioholdings pph
      JOIN papertradeportfolio ptp ON ptp.PaperPortfolioID = pph.PaperPortfolioID
      WHERE ${where.join(" AND ")}
    `;
      const [cntRows] = await db.query(countSql, params);
      const total = cntRows?.[0]?.total || 0;

      // ดึง holdings ตามจริง (ไม่คำนวณราคา/มูลค่าใด ๆ)
      const dataSql = `
      SELECT 
        pph.PaperHoldingID,
        pph.StockSymbol,
        pph.Quantity,
        pph.BuyPrice,
        pph.PaperPortfolioID
      FROM paperportfolioholdings pph
      JOIN papertradeportfolio ptp ON ptp.PaperPortfolioID = pph.PaperPortfolioID
      WHERE ${where.join(" AND ")}
      ORDER BY pph.StockSymbol
      LIMIT ? OFFSET ?
    `;

      const [rows] = await db.query(dataSql, [...params, limit, offset]);

      return res.status(200).json({
        message: "OK",
        data: rows, // [{PaperHoldingID, StockSymbol, Quantity, BuyPrice, PaperPortfolioID}, ...]
        pagination: {
          currentPage: page,
          totalPages: Math.ceil(total / limit),
          total,
          limit,
        },
      });
    } catch (err) {
      console.error("GET /api/admin/users/:userId/holdings-simple error:", err);
      return res.status(500).json({ error: "Internal server error" });
    }
  }
);

//=====================================================================================================//
// 										API ทั้งหมดสำหรับหน้า Dashboard
//=====================================================================================================//

/**
 * API: ดึงรายชื่อหุ้นทั้งหมดตามตลาด (สำหรับ Dropdown)
 * - รับค่า market จาก query parameter เช่น /api/stocks?market=Thailand
 */
app.get("/api/stocks", verifyToken, async (req, res) => {
  try {
    const { market } = req.query;

    if (!market) {
      return res
        .status(400)
        .json({ error: "Market query parameter is required." });
    }

    const validMarkets = ["Thailand", "America"];
    if (!validMarkets.includes(market)) {
      return res.status(400).json({
        error: "Invalid market specified. Use 'Thailand' or 'America'.",
      });
    }

    const sql = `
            SELECT StockSymbol, CompanyName 
            FROM Stock 
            WHERE Market = ? 
            ORDER BY StockSymbol ASC
        `;

    const [results] = await pool.promise().query(sql, [market]);

    res.status(200).json({
      message: `Successfully retrieved ${results.length} stocks for ${market}`,
      data: results,
    });
  } catch (error) {
    console.error("Internal server error in /api/stocks:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//=====================================================================================================//
// 										API: GET CHART DATA FOR A SPECIFIC STOCK (FIXED)
//=====================================================================================================//

app.get("/api/chart-data/:symbol", verifyToken, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = "1M" } = req.query;

    // กำหนดจำนวนข้อมูลที่จะดึงตามช่วงเวลา (เพิ่ม '1D' และ 'ALL' เข้ามา)
    const timeFrameLimits = {
      "1D": 1,
      "5D": 5,
      "1M": 22,
      "3M": 66,
      "6M": 132,
      "1Y": 252,
      ALL: null, // null หมายถึงไม่จำกัดจำนวน
    };

    const upperCaseTimeframe = timeframe.toUpperCase();
    if (!timeFrameLimits.hasOwnProperty(upperCaseTimeframe)) {
      return res.status(400).json({
        error:
          "Invalid timeframe. Use '1D', '5D', '1M', '3M', '6M', '1Y', or 'ALL'.",
      });
    }

    const limit = timeFrameLimits[upperCaseTimeframe];

    let sql;
    let params;

    // Logic สำหรับดึงข้อมูลย้อนหลัง (ใช้ได้กับทุก Timeframe ที่มี limit)
    if (limit !== null) {
      sql = `
                SELECT * FROM (
                    SELECT 
                        DATE_FORMAT(Date, '%Y-%m-%d') as date, 
                        ClosePrice 
                    FROM stockdetail 
                    WHERE StockSymbol = ? AND Volume != 0
                    ORDER BY Date DESC
                    LIMIT ?
                ) AS sub
                ORDER BY date ASC;
            `;
      params = [symbol, limit];
    }
    // Logic สำหรับดึงข้อมูลทั้งหมด ('ALL')
    else {
      sql = `
                SELECT 
                    DATE_FORMAT(Date, '%Y-%m-%d') as date, 
                    ClosePrice 
                FROM stockdetail 
                WHERE StockSymbol = ? AND Volume != 0
                ORDER BY Date ASC;
            `;
      params = [symbol];
    }

    const [results] = await pool.promise().query(sql, params);

    if (results.length === 0) {
      return res
        .status(404)
        .json({ message: `No historical data found for symbol ${symbol}.` });
    }

    res.status(200).json({
      message: `Successfully retrieved chart data for ${symbol}`,
      timeframe: timeframe,
      data: results,
    });
  } catch (error) {
    console.error("Internal server error in /api/chart-data:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// API: GET TOP 3 GAINERS AND LOSERS BY MARKET (STRICT: trading day only)
app.get("/api/market-movers", verifyToken, async (req, res) => {
  try {
    const { market } = req.query; // 'Thailand' หรือ 'America'

    if (!market || !["Thailand", "America"].includes(market)) {
      return res
        .status(400)
        .json({ error: "Invalid or missing market parameter." });
    }

    // 1) หา 'วันล่าสุดที่มีการเทรดจริง' ของ market นั้นๆ (Volume > 0) และให้ MySQL คืนเป็น DATE เลย
    const [latestDateRows] = await pool.promise().query(
      `
        SELECT DATE(MAX(sd.Date)) AS latestDate
        FROM stockdetail sd
        JOIN stock s ON sd.StockSymbol = s.StockSymbol
        WHERE s.Market = ? AND sd.Volume > 0
      `,
      [market]
    );

    const latestDate = latestDateRows?.[0]?.latestDate;

    if (!latestDate) {
      return res.status(200).json({
        message: "No trading day found for the specified market (Volume = 0).",
        data: { topGainers: [], topLosers: [] },
      });
    }

    // 2) Top 3 Gainers (เฉพาะวันล่าสุด และมีการเทรดจริง)
    const gainersSql = `
      SELECT s.StockSymbol, sd.ClosePrice, sd.Changepercen, sd.Volume
      FROM stockdetail sd
      JOIN stock s ON sd.StockSymbol = s.StockSymbol
      WHERE s.Market = ?
        AND DATE(sd.Date) = ?
        AND sd.Changepercen > 0
        AND sd.Volume > 0
      ORDER BY sd.Changepercen DESC
      LIMIT 3
    `;
    const [topGainers] = await pool
      .promise()
      .query(gainersSql, [market, latestDate]);

    // 3) Top 3 Losers (เฉพาะวันล่าสุด และมีการเทรดจริง)
    const losersSql = `
      SELECT s.StockSymbol, sd.ClosePrice, sd.Changepercen, sd.Volume
      FROM stockdetail sd
      JOIN stock s ON sd.StockSymbol = s.StockSymbol
      WHERE s.Market = ?
        AND DATE(sd.Date) = ?
        AND sd.Changepercen < 0
        AND sd.Volume > 0
      ORDER BY sd.Changepercen ASC
      LIMIT 3
    `;
    const [topLosers] = await pool
      .promise()
      .query(losersSql, [market, latestDate]);

    return res.status(200).json({
      message: `Successfully retrieved market movers for ${market}`,
      date: latestDate, // YYYY-MM-DD จาก MySQL โดยตรง
      data: { topGainers, topLosers },
    });
  } catch (error) {
    console.error("Internal server error in /api/market-movers:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

//=====================================================================================================//
//  API: AI TRADE MONITOR (ADMIN) — JOIN กับ user เพื่อเอา Username แทน UserID
//=====================================================================================================//

app.get("/api/admin/ai-trades", verifyToken, verifyAdmin, async (req, res) => {
  const db = pool.promise(); // ✅ ใช้ promise wrapper จาก pool เดิม
  try {
    // pagination
    const page = Math.max(parseInt(req.query.page) || 1, 1);
    const limit = Math.min(Math.max(parseInt(req.query.limit) || 20, 1), 200);
    const offset = (page - 1) * limit;

    // optional filters
    const { userId, symbol, action, date_from, date_to } = req.query;

    // allowlist orderBy กัน SQL injection
    const ORDERABLE = new Set([
      "PaperTradeID",
      "TradeType",
      "Quantity",
      "Price",
      "TradeDate",
      "Username",
      "StockSymbol",
    ]);
    const orderBy = ORDERABLE.has(req.query.orderBy)
      ? req.query.orderBy
      : "TradeDate";
    const order =
      String(req.query.order || "DESC").toUpperCase() === "ASC"
        ? "ASC"
        : "DESC";

    // WHERE clause
    const where = [];
    const params = [];

    if (userId) {
      where.push("pt.UserID = ?");
      params.push(userId);
    }
    if (symbol) {
      where.push("pt.StockSymbol = ?");
      params.push(symbol);
    }
    if (action) {
      where.push("pt.TradeType = ?");
      params.push(action);
    }
    if (date_from) {
      where.push("DATE(pt.TradeDate) >= ?");
      params.push(date_from);
    }
    if (date_to) {
      where.push("DATE(pt.TradeDate) <= ?");
      params.push(date_to);
    }

    const whereClause = where.length ? `WHERE ${where.join(" AND ")}` : "";

    // นับจำนวนทั้งหมด
    const countSql = `
      SELECT COUNT(*) AS total
      FROM trademine.papertrade pt
      JOIN trademine.user u ON pt.UserID = u.UserID
      ${whereClause}
    `;
    const [countRows] = await db.query(countSql, params);
    const totalTrades = countRows?.[0]?.total ?? 0;

    // ดึงข้อมูล
    const dataSql = `
      SELECT
        pt.PaperTradeID,
        pt.TradeType,
        pt.Quantity,
        pt.Price,
        pt.TradeDate,
        u.Username AS Username,
        pt.StockSymbol
      FROM trademine.papertrade pt
      JOIN trademine.user u ON pt.UserID = u.UserID
      ${whereClause}
      ORDER BY ${orderBy} ${order}
      LIMIT ? OFFSET ?
    `;
    const [rows] = await db.query(dataSql, [...params, limit, offset]);

    return res.status(200).json({
      message: "OK",
      data: rows, // ตอนนี้ได้ Username แล้วแทน UserID
      pagination: {
        currentPage: page,
        totalPages: Math.ceil(totalTrades / limit),
        totalTrades,
        limit,
      },
    });
  } catch (err) {
    console.error("Internal server error /api/admin/ai-trades:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// === MARKET TREND (2 ENDPOINTS) ===

// 1) SYMBOLS by market (dropdown)
app.get("/api/market-trend/symbols", verifyToken, async (req, res) => {
  const db = pool.promise();
  try {
    const market = (req.query.market || "").trim();
    const limit = Math.min(Math.max(parseInt(req.query.limit) || 500, 1), 2000);
    if (!market) return res.status(400).json({ error: "market is required" });

    // ถ้ามีตาราง Stock ให้ใช้แบบนี้ (มี CompanyName & Market)
    const [rows] = await db.query(
      `
      SELECT s.StockSymbol, s.CompanyName, s.Market, MAX(sd.Date) AS newestDate
      FROM Stock s
      LEFT JOIN trademine.stockdetail sd
        ON sd.StockSymbol = s.StockSymbol
      WHERE s.Market = ?
      GROUP BY s.StockSymbol, s.CompanyName, s.Market
      ORDER BY s.StockSymbol
      LIMIT ?
      `,
      [market, limit]
    );

    res.status(200).json({ message: "OK", data: rows });
  } catch (err) {
    console.error("symbols error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// 2) DATA (latest + historical)
// โหมด A: ?symbol=PTT&limit=66   -> N วันล่าสุด (distinct date)
// โหมด B: ?symbol=PTT&from=YYYY-MM-DD&to=YYYY-MM-DD -> ตามช่วงวันที่
app.get("/api/market-trend/data", verifyToken, async (req, res) => {
  const db = pool.promise();
  try {
    const symbol = (req.query.symbol || "").trim().toUpperCase();
    if (!symbol) return res.status(400).json({ error: "symbol is required" });

    const from = (req.query.from || "").trim();
    const to = (req.query.to || "").trim();
    const limit = Math.min(Math.max(parseInt(req.query.limit) || 66, 1), 1000);

    // latest แถวเดียว (ไม่กรอง Volume)
    const [latestRows] = await db.query(
      `
      SELECT 
        StockDetailID, StockSymbol, Date,
        OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
      FROM trademine.stockdetail
      WHERE StockSymbol = ?
      ORDER BY Date DESC
      LIMIT 1
      `,
      [symbol]
    );
    const latest = latestRows[0] || null;

    // historical series
    let series = [];
    if (from && to) {
      // โหมดช่วงวันที่
      const [rows] = await db.query(
        `
        SELECT DATE_FORMAT(Date,'%Y-%m-%d') AS date,
               OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
        FROM trademine.stockdetail
        WHERE StockSymbol = ?
          AND Date BETWEEN ? AND ?
        ORDER BY Date ASC
        `,
        [symbol, from, to]
      );
      series = rows;
    } else {
      // โหมดจำนวนแท่งล่าสุด (ใช้ DISTINCT date กันวันหยุด)
      const [rows] = await db.query(
        `
        SELECT 
          DATE_FORMAT(s.Date,'%Y-%m-%d') AS date,
          s.OpenPrice, s.HighPrice, s.LowPrice, s.ClosePrice, s.Volume
        FROM trademine.stockdetail s
        JOIN (
          SELECT d FROM (
            SELECT DISTINCT Date AS d
            FROM trademine.stockdetail
            WHERE StockSymbol = ?
            ORDER BY Date DESC
            LIMIT ?
          ) dd
        ) lastd ON lastd.d = s.Date
        WHERE s.StockSymbol = ?
        ORDER BY s.Date ASC
        `,
        [symbol, limit, symbol]
      );
      series = rows;
    }

    if (!latest && series.length === 0) {
      return res.status(404).json({ error: "No data" });
    }

    res.status(200).json({
      message: "OK",
      symbol,
      latest,
      series,
    });
  } catch (err) {
    console.error("data error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/model-performance", async (req, res) => {
  const { symbol, start, end } = req.query;
  const sql = `
        SELECT Date, ClosePrice, 
               PredictionClose_LSTM, PredictionClose_GRU, PredictionClose_Ensemble,
               PredictionTrend_LSTM, PredictionTrend_GRU, PredictionTrend_Ensemble
        FROM stockdetail
        WHERE StockSymbol = ? AND Date BETWEEN ? AND ?
        ORDER BY Date
    `;
  const [rows] = await db.query(sql, [symbol, start, end]);

  const calcRMSE = (actual, predicted) => {
    let mse =
      actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0) /
      actual.length;
    return Math.sqrt(mse);
  };

  const calcMAPE = (actual, predicted) => {
    let ape =
      actual.reduce(
        (sum, val, i) => sum + Math.abs((val - predicted[i]) / val),
        0
      ) / actual.length;
    return ape * 100;
  };

  const calcTrendAccuracy = (actual, predicted) => {
    let correct = actual.reduce((count, val, i, arr) => {
      if (i === 0) return count;
      let actualTrend = val > arr[i - 1] ? "UP" : "DOWN";
      let predTrend = predicted[i] > predicted[i - 1] ? "UP" : "DOWN";
      return count + (actualTrend === predTrend ? 1 : 0);
    }, 0);
    return (correct / (actual.length - 1)) * 100;
  };

  const actual = rows.map((r) => r.ClosePrice);

  const performance = {
    LSTM: {
      RMSE: calcRMSE(
        actual,
        rows.map((r) => r.PredictionClose_LSTM)
      ),
      MAPE: calcMAPE(
        actual,
        rows.map((r) => r.PredictionClose_LSTM)
      ),
      TrendAccuracy: calcTrendAccuracy(
        actual,
        rows.map((r) => r.PredictionClose_LSTM)
      ),
    },
    GRU: {
      RMSE: calcRMSE(
        actual,
        rows.map((r) => r.PredictionClose_GRU)
      ),
      MAPE: calcMAPE(
        actual,
        rows.map((r) => r.PredictionClose_GRU)
      ),
      TrendAccuracy: calcTrendAccuracy(
        actual,
        rows.map((r) => r.PredictionClose_GRU)
      ),
    },
    Ensemble: {
      RMSE: calcRMSE(
        actual,
        rows.map((r) => r.PredictionClose_Ensemble)
      ),
      MAPE: calcMAPE(
        actual,
        rows.map((r) => r.PredictionClose_Ensemble)
      ),
      TrendAccuracy: calcTrendAccuracy(
        actual,
        rows.map((r) => r.PredictionClose_Ensemble)
      ),
    },
  };

  res.json({ data: rows, performance });
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// รันทุก
cron.schedule(
  "0 */2 * * *",
  async () => {
    try {
      console.log("⏰ Cron: pushing today news...");
      // เรียกฟังก์ชันผ่าน HTTP ภายในก็ได้ แต่ประหยัดกว่าถ้า refactor handler เป็นฟังก์ชัน
      // ที่นี่ขอยิงผ่าน axios ไปที่ตัวเอง (ถ้า server รับจากภายนอกได้)
      await axios.get(
        "http://localhost:" +
          (process.env.PORT || 3000) +
          "/api/news-notifications"
      );
      console.log("✅ Cron done");
    } catch (e) {
      console.error("❌ Cron error:", e.message || e);
    }
  },
  { timezone: "Asia/Bangkok" }
);

const AUTOTRADE_URL =
  process.env.AUTOTRADE_URL || `http://localhost:3000/api/autoTrade/run`;
let isRunning = false;

cron.schedule(
  "0 10,22 * * *", // ทุกวัน 10:00 และ 22:00 (4 ทุ่ม)
  async () => {
    if (isRunning) {
      console.log("⏭️  Cron autotrade: previous run still in progress, skip.");
      return;
    }
    isRunning = true;
    const start = Date.now();

    try {
      console.log("⏰ Cron autotrade: kicking /api/autoTrade/run ...");
      const resp = await axios.post(
        AUTOTRADE_URL,
        {},
        { timeout: 5 * 60 * 1000 }
      );

      const portfolios = resp?.data?.portfolios?.length ?? 0;
      console.log(
        `✅ Cron autotrade done in ${
          Date.now() - start
        }ms (portfolios=${portfolios})`
      );
    } catch (e) {
      const status = e?.response?.status;
      console.error(
        "❌ Cron autotrade error:",
        status ? `HTTP ${status}` : "",
        e?.message || e
      );
    } finally {
      isRunning = false;
    }
  },
  { timezone: "Asia/Bangkok" }
);
