const express = require("express");
const bodyParser = require("body-parser");
const mysql = require("mysql2");
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
const JWT_SECRET = process.env.JWT_SECRET;
const app = express();
const { PythonShell } = require('python-shell');

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors()); // Enable CORS


// Create Connection Pool
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

  // ฟังก์ชันสำหรับตรวจสอบ JWT token
const verifyToken = (req, res, next) => {
  const token = req.headers['authorization'];

  if (!token) {
    return res.status(403).json({ message: "Token is required" });
  }

  // ตัดคำว่า "Bearer" ออกจาก token
  const bearerToken = token.split(' ')[1];

  jwt.verify(bearerToken, JWT_SECRET, (err, decoded) => {
    if (err) {
      return res.status(401).json({ message: "Invalid token" });
    }
    req.userId = decoded.id; // เก็บ userId จาก token ใน request object
    next(); // ส่งต่อให้กับ middleware หรือ route handler ถัดไป
  });
};

module.exports = verifyToken; // เพื่อให้สามารถนำไปใช้ในไฟล์อื่นได้

// Storage configuration for profile picture upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
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

//User-Register-Email
app.post("/api/register/email", async (req, res) => {
  try {
      const { email } = req.body;

      if (!email) {
          return res.status(400).json({ error: "กรุณากรอกอีเมล" });
      }

      pool.query("SELECT * FROM User WHERE Email = ?", [email], (err, results) => {
          if (err) {
              console.error("Database error during email check:", err);
              return res.status(500).json({ error: "Database error during email check" });
          }

          if (results.length > 0) {
              const user = results[0];

              // ถ้า Email นี้เคยลงทะเบียนแล้วและเป็น Active
              if (user.Status === "active" && user.Password) {
                  return res.status(400).json({ error: "อีเมลนี้ถูกลงทะเบียนแล้ว" });
              }

              // ถ้าเคยสมัครแต่เป็น deactivated ให้เปิดใช้งานอีกครั้ง
              if (user.Status === "deactivated") {
                  pool.query("UPDATE User SET Status = 'active' WHERE Email = ?", [email]);
                  return res.status(200).json({ message: "บัญชีของคุณถูกเปิดใช้งานอีกครั้ง" });
              }
          }

          // **สร้าง OTP และกำหนดเวลา Expiry**
          const otp = generateOtp();
          const expiresAt = new Date(Date.now() + 3 * 60 * 1000); // OTP หมดอายุใน 3 นาที
          const createdAt = new Date(Date.now());

          // **เพิ่มข้อมูล User ใหม่หากยังไม่มี**
          pool.query(
              "INSERT INTO User (Email, Username, Password, Status) VALUES (?, '', '', 'pending') ON DUPLICATE KEY UPDATE Status = 'pending'",
              [email],
              (err) => {
                  if (err) {
                      console.error("Database error during User insertion or update:", err);
                      return res.status(500).json({ error: "Database error during User insertion or update" });
                  }

                  // **บันทึก OTP เชื่อมกับ Email แทน UserID**
// ดึง UserID จาก Email ก่อน
                pool.query(
                  "SELECT UserID FROM User WHERE Email = ?",
                  [email],
                  (err, results) => {
                      if (err) {
                          console.error("Error fetching UserID:", err);
                          return res.status(500).json({ error: "Database error fetching UserID" });
                      }

                      if (results.length === 0) {
                          return res.status(404).json({ error: "User not found" });
                      }

                      const userId = results[0].UserID;

                      // แทรก OTP โดยใช้ UserID แทน Email
                      pool.query(
                          "INSERT INTO OTP (OTP_Code, Created_At, Expires_At, UserID) VALUES (?, ?, ?, ?) ON DUPLICATE KEY UPDATE OTP_Code = ?, Created_At = ?, Expires_At = ?",
                          [otp, createdAt, expiresAt, userId, otp, createdAt, expiresAt],
                          (err) => {
                              if (err) {
                                  console.error("Error during OTP insertion:", err);
                                  return res.status(500).json({ error: "Database error during OTP insertion" });
                              }

                              console.log("OTP inserted successfully");
                              sendOtpEmail(email, otp, (error) => {
                                  if (error) return res.status(500).json({ error: "Error sending OTP email" });
                                  res.status(200).json({ message: "OTP ถูกส่งไปยังอีเมลของคุณ" });
                              });
                          }
                      );
                  }
                );

              }
          );
      });
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
    pool.query("SELECT UserID FROM User WHERE Email = ?", [email], (err, userResults) => {
      if (err) return res.status(500).json({ error: "Database error" });

      if (userResults.length === 0) {
        return res.status(404).json({ error: "ไม่พบ Email ในระบบ" });
      }

      const userId = userResults[0].UserID;

      // ค้นหา OTP ในฐานข้อมูลโดยใช้ UserID และ OTP
      pool.query("SELECT * FROM OTP WHERE UserID = ? AND OTP_Code = ?", [userId, otp], (err, otpResults) => {
        if (err) return res.status(500).json({ error: "Database error" });

        // ถ้าไม่พบ OTP ในฐานข้อมูล
        if (otpResults.length === 0) {
          return res.status(400).json({ error: "OTP ไม่ถูกต้อง" });
        }

        // ตรวจสอบว่า OTP ยังไม่หมดอายุ
        const { Expires_At } = otpResults[0];
        if (new Date() > new Date(Expires_At)) {
          return res.status(400).json({ error: "OTP หมดอายุ" });
        }

        // ถ้า OTP ถูกต้องและไม่หมดอายุ
        res.status(200).json({ message: "OTP ถูกต้อง คุณสามารถตั้งรหัสผ่านได้" });
      });
    });
  } catch (error) {
    res.status(500).json({ error: "Internal server error" });
  }
});

// User-Set-Password (อัปเดตใหม่)
app.post("/api/register/set-password", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Email และ Password ต้องถูกต้อง" });
    }

    const hash = await bcrypt.hash(password, 10);

    // ดึง UserID ก่อนอัปเดตรหัสผ่าน
    pool.query("SELECT UserID FROM User WHERE Email = ?", [email], (err, results) => {
      if (err) {
        console.error("Error fetching UserID:", err);
        return res.status(500).json({ error: "Database error fetching UserID" });
      }

      if (results.length === 0) {
        return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });
      }

      const userId = results[0].UserID;

      // อัปเดตรหัสผ่านในตาราง User แต่ยังไม่เปลี่ยนเป็น 'active'
      pool.query(
        "UPDATE User SET Password = ?, Status = 'pending' WHERE UserID = ?",
        [hash, userId],
        (err, results) => {
          if (err) {
            console.error("Database error during User update:", err);
            return res.status(500).json({ error: "Database error during User update" });
          }

          if (results.affectedRows === 0) {
            return res.status(404).json({ error: "ไม่สามารถอัปเดตรหัสผ่านได้" });
          }

          // ลบ OTP ที่เกี่ยวข้องกับ UserID
          pool.query("DELETE FROM OTP WHERE UserID = ?", [userId], (err) => {
            if (err) {
              console.error("Error during OTP deletion:", err);
              return res.status(500).json({ error: "Error during OTP deletion" });
            }

            // **สร้าง Token และส่งกลับ**
            const token = jwt.sign({ id: userId }, JWT_SECRET, { expiresIn: "7d" });

            res.status(200).json({
              message: "รหัสผ่านถูกตั้งเรียบร้อยแล้ว กรุณาตั้งค่าโปรไฟล์",
              token: token
            });
          });
        }
      );
    });
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
    pool.query("SELECT UserID FROM User WHERE Email = ?", [email], (err, userResults) => {
      if (err) return res.status(500).json({ error: "Database error during user lookup" });
      if (userResults.length === 0) return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });

      const userId = userResults[0].UserID; // ดึง UserID ที่แท้จริง

      // ค้นหา OTP ที่มีอยู่ ถ้ามีให้อัปเดต ถ้าไม่มีให้แทรกใหม่
      pool.query("SELECT * FROM OTP WHERE UserID = ?", [userId], (err, otpResults) => {
        if (err) return res.status(500).json({ error: "Database error during OTP check" });

        if (otpResults.length > 0) {
          // ถ้ามี OTP อยู่แล้ว อัปเดตข้อมูลใหม่
          pool.query(
            "UPDATE OTP SET OTP_Code = ?, Expires_At = ? WHERE UserID = ?",
            [newOtp, newExpiresAt, userId],
            (err) => {
              if (err) return res.status(500).json({ error: "Database error during OTP update" });

              // ส่ง OTP ไปยังอีเมลของผู้ใช้
              sendOtpEmail(email, newOtp, (error) => {
                if (error) return res.status(500).json({ error: "Error sending OTP email" });
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
              if (err) return res.status(500).json({ error: "Database error during OTP insertion" });

              // ส่ง OTP ไปยังอีเมลของผู้ใช้
              sendOtpEmail(email, newOtp, (error) => {
                if (error) return res.status(500).json({ error: "Error sending OTP email" });
                res.status(200).json({ message: "OTP ถูกส่งใหม่แล้ว" });
              });
            }
          );
        }
      });
    });
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
    const userCheckSql = "SELECT UserID FROM User WHERE Email = ? AND Password IS NOT NULL AND Status = 'active'";
    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err) return res.status(500).json({ error: "Database error during email check", details: err });

      if (userResults.length === 0) {
        return res.status(400).json({ error: "Email not found or inactive" });
      }

      const userId = userResults[0].UserID; // ดึง UserID จาก Email

      // สร้าง OTP ใหม่
      const otp = generateOtp();
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // OTP หมดอายุใน 10 นาที

      // ตรวจสอบว่า User มี OTP อยู่แล้วหรือไม่
      pool.query("SELECT * FROM OTP WHERE UserID = ?", [userId], (err, otpResults) => {
        if (err) return res.status(500).json({ error: "Database error during OTP check", details: err });

        if (otpResults.length > 0) {
          // ถ้ามี OTP อยู่แล้ว ให้ทำการอัปเดต
          const updateOtpSql = "UPDATE OTP SET OTP_Code = ?, Expires_At = ?, Created_At = NOW() WHERE UserID = ?";
          pool.query(updateOtpSql, [otp, expiresAt, userId], (err) => {
            if (err) return res.status(500).json({ error: "Database error during OTP update", details: err });

            sendOtpEmail(email, otp, (error) => {
              if (error) return res.status(500).json({ error: "Error sending OTP email" });
              res.status(200).json({ message: "OTP sent to email" });
            });
          });
        } else {
          // ถ้ายังไม่มี OTP ให้เพิ่มเข้าไป
          const saveOtpSql = "INSERT INTO OTP (UserID, OTP_Code, Expires_At, Created_At) VALUES (?, ?, ?, NOW())";
          pool.query(saveOtpSql, [userId, otp, expiresAt], (err) => {
            if (err) return res.status(500).json({ error: "Database error during OTP save", details: err });

            sendOtpEmail(email, otp, (error) => {
              if (error) return res.status(500).json({ error: "Error sending OTP email" });
              res.status(200).json({ message: "OTP sent to email" });
            });
          });
        }
      });
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
    pool.query("SELECT UserID FROM User WHERE Email = ?", [email], (err, userResults) => {
      if (err) return res.status(500).json({ error: "Database error during user lookup" });
      if (userResults.length === 0) return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });

      const userId = userResults[0].UserID;

      // ค้นหา OTP ในตาราง OTP โดยใช้ UserID และ OTP ที่ส่งมา
      pool.query(
        "SELECT OTP_Code, Expires_At FROM OTP WHERE UserID = ? AND OTP_Code = ?",
        [userId, otp],
        (err, results) => {
          if (err) return res.status(500).json({ error: "Database error during OTP verification" });

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
          res.status(200).json({ message: "OTP ถูกต้อง คุณสามารถตั้งรหัสผ่านใหม่ได้" });
        }
      );
    });
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
    pool.query("SELECT UserID FROM User WHERE Email = ?", [email], (err, userResults) => {
      if (err) return res.status(500).json({ error: "Database error during user lookup" });
      if (userResults.length === 0) return res.status(404).json({ error: "ไม่พบบัญชีที่ใช้ Email นี้" });

      const userId = userResults[0].UserID;

      // อัปเดตรหัสผ่านในตาราง User
      pool.query(
        "UPDATE User SET Password = ?, Status = 'active' WHERE Email = ?",
        [hashedPassword, email],
        (err) => {
          if (err) {
            console.error("Database error during password update:", err);
            return res.status(500).json({ error: "Database error during password update" });
          }

          // ลบ OTP ที่เกี่ยวข้องกับ UserID จากตาราง OTP
          pool.query("DELETE FROM OTP WHERE UserID = ?", [userId], (err) => {
            if (err) {
              console.error("Error during OTP deletion:", err);
              return res.status(500).json({ error: "Error during OTP deletion" });
            }

            res.status(200).json({ message: "รหัสผ่านถูกตั้งใหม่เรียบร้อยแล้ว" });
          });
        }
      );
    });
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
      return res.status(400).json({ error: "Email is required" });
    }

    // ตรวจสอบว่ามี Email นี้ในตาราง User
    const userCheckSql = "SELECT UserID FROM User WHERE Email = ?";
    pool.query(userCheckSql, [email], (err, userResults) => {
      if (err) return res.status(500).json({ error: "Database error during user lookup" });
      if (userResults.length === 0) return res.status(404).json({ error: "User not found" });

      const userId = userResults[0].UserID; // ดึง UserID จาก Email

      // สร้าง OTP ใหม่
      const otp = generateOtp();
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // OTP หมดอายุใน 10 นาที

      // ตรวจสอบว่า User มี OTP อยู่แล้วหรือไม่
      pool.query("SELECT * FROM OTP WHERE UserID = ?", [userId], (err, otpResults) => {
        if (err) return res.status(500).json({ error: "Database error during OTP lookup" });

        if (otpResults.length > 0) {
          // ถ้ามี OTP อยู่แล้ว ให้ทำการอัปเดต
          const updateOtpSql = "UPDATE OTP SET OTP_Code = ?, Expires_At = ? WHERE UserID = ?";
          pool.query(updateOtpSql, [otp, expiresAt, userId], (err) => {
            if (err) return res.status(500).json({ error: "Database error during OTP update" });

            sendOtpEmail(email, otp, (error) => {
              if (error) return res.status(500).json({ error: "Error sending OTP email" });
              res.status(200).json({ message: "New OTP sent to email" });
            });
          });
        } else {
          // ถ้ายังไม่มี OTP ให้เพิ่มเข้าไป
          const insertOtpSql = "INSERT INTO OTP (UserID, OTP_Code, Created_At, Expires_At) VALUES (?, ?, NOW(), ?)";
          pool.query(insertOtpSql, [userId, otp, expiresAt], (err) => {
            if (err) return res.status(500).json({ error: "Database error during OTP insert" });

            sendOtpEmail(email, otp, (error) => {
              if (error) return res.status(500).json({ error: "Error sending OTP email" });
              res.status(200).json({ message: "New OTP sent to email" });
            });
          });
        }
      });
    });
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


//Login
app.post("/api/login", async (req, res) => {
  try {
    const { email, password, googleId } = req.body;

    // รับ IP Address ของผู้ใช้
    const ipAddress = req.headers["x-forwarded-for"] || req.connection.remoteAddress;

    // ค้นหาผู้ใช้จากฐานข้อมูล
    pool.query("SELECT * FROM User WHERE Email = ?", [email], (err, results) => {
      if (err) return res.status(500).json({ error: "Database error" });

      if (results.length === 0) {
        return res.status(404).json({ message: "ไม่พบบัญชีนี้" });
      }

      const user = results[0];

      // ตรวจสอบว่าสถานะบัญชีเป็น Active หรือไม่
      if (user.Status !== "active") {
        return res.status(403).json({ message: "บัญชีถูกระงับการใช้งาน" });
      }

      if (googleId) {
        // ค้นหาผู้ใช้จาก GoogleID
        pool.query("SELECT * FROM User WHERE GoogleID = ?", [googleId], (err, results) => {
          if (err) {
            return res.status(500).json({ message: "เกิดข้อผิดพลาดในระบบ" });
          }
      
          if (results.length > 0) {
            // ผู้ใช้เคยล็อกอินด้วย Google แล้ว
            const user = results[0];
      
            // สร้าง JWT Token
            const token = jwt.sign({ id: user.UserID, role: user.Role }, JWT_SECRET);
      
            // อัปเดตข้อมูลล็อกอินล่าสุด
            pool.query("UPDATE User SET LastLogin = NOW(), LastLoginIP = ? WHERE UserID = ?", [ipAddress, user.UserID]);
      
            return res.status(200).json({
              message: "เข้าสู่ระบบด้วย Google สำเร็จ",
              token,
              user: {
                id: user.UserID,
                email: user.Email,
                username: user.Username,
              },
            });
          } else {
            // ล็อกอินครั้งแรก → ลงทะเบียนบัญชีใหม่
            pool.query(
              "INSERT INTO User (GoogleID, Email, Username, Role, LastLogin, LastLoginIP) VALUES (?, ?, ?, 'user', NOW(), ?)",
              [googleId, googleEmail, googleUsername, ipAddress],
              (err, result) => {
                if (err) {
                  return res.status(500).json({ message: "เกิดข้อผิดพลาดในการสร้างบัญชี" });
                }
      
                const newUserId = result.insertId;
                const token = jwt.sign({ id: newUserId, role: "user" }, JWT_SECRET);
      
                return res.status(201).json({
                  message: "สมัครสมาชิกและเข้าสู่ระบบด้วย Google สำเร็จ",
                  token,
                  user: {
                    id: newUserId,
                    email: googleEmail,
                    username: googleUsername,
                  },
                });
              }
            );
          }
        });
      }
      

      // 📌 ถ้าผู้ใช้ล็อกอินด้วย Email + Password
      if (!password) {
        return res.status(400).json({ message: "กรุณากรอกรหัสผ่าน" });
      }

      // ถ้าผู้ใช้ลงทะเบียนด้วย Google ให้แจ้งให้ใช้ Google Login
      if (user.GoogleID !== null) {
        return res.status(400).json({ message: "กรุณาเข้าสู่ระบบด้วย Google" });
      }

      // ตรวจสอบจำนวนครั้งที่ล็อกอินผิดพลาด
      if (user.FailedAttempts >= 5 && user.LastFailedAttempt) {
        const now = Date.now();
        const timeSinceLastAttempt = now - new Date(user.LastFailedAttempt).getTime();
        if (timeSinceLastAttempt < 300000) { // 5 นาที
          return res.status(429).json({ message: "คุณล็อกอินผิดพลาดหลายครั้ง โปรดลองอีกครั้งใน 5 นาที" });
        }
      }

      // ตรวจสอบรหัสผ่าน (bcrypt)
      bcrypt.compare(password, user.Password, (err, isMatch) => {
        if (err) return res.status(500).json({ error: "Error comparing passwords" });

        if (!isMatch) {
          // บันทึกความล้มเหลวของการล็อกอิน
          pool.query("UPDATE User SET FailedAttempts = FailedAttempts + 1, LastFailedAttempt = NOW() WHERE UserID = ?", [user.UserID]);
          return res.status(401).json({ message: "อีเมลหรือรหัสผ่านไม่ถูกต้อง" });
        }

        // รีเซ็ตจำนวนครั้งที่ล็อกอินผิดพลาด
        pool.query("UPDATE User SET FailedAttempts = 0, LastLogin = NOW(), LastLoginIP = ? WHERE UserID = ?", [ipAddress, user.UserID]);

        // สร้าง JWT Token
        const token = jwt.sign({ id: user.UserID, role: user.Role }, JWT_SECRET, { expiresIn: "7d" });

        // ส่งข้อมูลกลับไปให้ผู้ใช้
        res.status(200).json({
          message: "เข้าสู่ระบบสำเร็จ",
          token,
          user: {
            id: user.UserID,
            email: user.Email,
            username: user.Username,
          },
        });
      });
    });
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Set Profile และ Login อัตโนมัติหลังจากตั้งโปรไฟล์เสร็จ
app.post("/api/set-profile", verifyToken, upload.single('picture'), (req, res) => {
  const { newUsername, birthday } = req.body;
  const userId = req.userId; // รับ UserID จาก token
  const picture = req.file ? `/uploads/${req.file.filename}` : null;

  // ตรวจสอบว่า newUsername, picture, และ birthday ถูกส่งมาหรือไม่
  if (!newUsername || !picture || !birthday) {
    return res.status(400).json({ message: "New username, picture, and birthday are required" });
  }

  // แปลงวันที่จาก DD/MM/YYYY เป็น YYYY-MM-DD
  const birthdayParts = birthday.split('/');
  const formattedBirthday = `${birthdayParts[2]}-${birthdayParts[1]}-${birthdayParts[0]}`;

  // อัปเดตโปรไฟล์ของผู้ใช้ และเปลี่ยนสถานะเป็น Active
  const updateProfileQuery = "UPDATE User SET Username = ?, ProfileImageURL = ?, Birthday = ?, Status = 'active' WHERE UserID = ?";
  pool.query(updateProfileQuery, [newUsername, picture, formattedBirthday, userId], (err) => {
    if (err) {
      console.error("Error updating profile: ", err);
      return res.status(500).json({ message: "Error updating profile" });
    }

    // ดึงข้อมูลผู้ใช้เพื่อนำไปสร้าง Token
    pool.query("SELECT UserID, Email, Username, ProfileImageURL FROM User WHERE UserID = ?", [userId], (err, userResults) => {
      if (err) {
        console.error("Database error fetching user data:", err);
        return res.status(500).json({ message: "Error fetching user data" });
      }

      if (userResults.length === 0) {
        return res.status(404).json({ message: "User not found after profile update" });
      }

      const user = userResults[0];

      // สร้าง JWT Token
      const token = jwt.sign({ id: user.UserID }, JWT_SECRET, { expiresIn: "7d" });

      return res.status(200).json({
        message: "Profile set successfully. You are now logged in.",
        token,
        user: {
          id: user.UserID,
          email: user.Email,
          username: user.Username,
          profileImage: user.ProfileImageURL,
        },
      });
    });
  });
});


// Login with Google * ยังไม่ได้เช็คบน Postman
app.post("/api/google-signin", async (req, res) => {
  try {
    const { googleId, email } = req.body;

    if (!googleId || !email) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    // Check if GoogleID already exists in the User table with 'active' or 'deactivated' status
    const checkGoogleIdSql = "SELECT * FROM User WHERE GoogleID = ? AND (Status = 'active' OR Status = 'deactivated')";
    pool.query(checkGoogleIdSql, [googleId], (err, googleIdResults) => {
      if (err) throw new Error("Database error during Google ID check");

      if (googleIdResults.length > 0) {
        const user = googleIdResults[0];

        // Reactivate user if status is 'deactivated'
        if (user.Status === "deactivated") {
          const reactivateSql = "UPDATE User SET Status = 'active', Email = ? WHERE GoogleID = ?";
          pool.query(reactivateSql, [email, googleId], (err) => {
            if (err) throw new Error("Database error during user reactivation");

            const token = jwt.sign({ id: user.UserID }, JWT_SECRET, { expiresIn: "7d" });
            return res.json({
              message: "User reactivated and authenticated successfully",
              token,
              user: {
                id: user.UserID,
                email: user.Email,
                username: user.Username,
                google_id: user.GoogleID,
                status: 'active',
              },
            });
          });
        } else {
          // If the user is already active, update email if necessary
          const updateSql = "UPDATE User SET Email = ? WHERE GoogleID = ?";
          pool.query(updateSql, [email, googleId], (err) => {
            if (err) throw new Error("Database error during user update");

            const token = jwt.sign({ id: user.UserID }, JWT_SECRET, { expiresIn: "7d" });
            return res.json({
              message: "User information updated successfully",
              token,
              user: {
                id: user.UserID,
                email: user.Email,
                username: user.Username,
                google_id: user.GoogleID,
                status: user.Status,
              },
            });
          });
        }
      } else {
        // Check if the email is already registered with another account
        const checkEmailSql = "SELECT * FROM User WHERE Email = ? AND Status = 'active'";
        pool.query(checkEmailSql, [email], (err, emailResults) => {
          if (err) throw new Error("Database error during email check");
          if (emailResults.length > 0) {
            return res.status(409).json({
              error: "Email already registered with another account",
            });
          }

          // If the user is not registered, create a new user with GoogleID, email, and status
          const insertSql = "INSERT INTO User (GoogleID, Email, Username, Status) VALUES (?, ?, '', 'active')";
          pool.query(insertSql, [googleId, email], (err, result) => {
            if (err) throw new Error("Database error during user insertion");

            const newUserId = result.insertId;
            const newUserSql = "SELECT * FROM User WHERE UserID = ?";
            pool.query(newUserSql, [newUserId], (err, newUserResults) => {
              if (err) throw new Error("Database error during new user fetch");

              const newUser = newUserResults[0];
              const token = jwt.sign({ id: newUser.UserID }, JWT_SECRET, { expiresIn: "7d" });

              return res.status(201).json({
                message: "User registered and authenticated successfully",
                token,
                user: {
                  id: newUser.UserID,
                  email: newUser.Email,
                  username: newUser.Username,
                  google_id: newUser.GoogleID,
                  status: newUser.Status,
                },
              });
            });
          });
        });
      }
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---- Search ---- //

app.get("/api/search", (req, res) => {
  const { query } = req.query;

  if (!query) {
    return res.status(400).json({ error: "Search query is required" });
  }

  // Trim the query และแปลงให้เป็นตัวพิมพ์เล็ก
  const searchValue = `%${query.trim().toLowerCase()}%`;

  // SQL query เพื่อค้นหาข้อมูลจาก Stock และ StockDetail
  const searchSql = `
    SELECT 
        s.StockSymbol, 
        s.Market, 
        s.MarketCap, 
        s.CompanyName, 
        sd.Date, 
        sd.ClosePrice
    FROM Stock s
    LEFT JOIN StockDetail sd ON s.StockSymbol = sd.StockSymbol
    WHERE LOWER(s.StockSymbol) LIKE ? 
       OR LOWER(s.CompanyName) LIKE ?
    ORDER BY sd.Date DESC; 
  `;

  pool.query(searchSql, [searchValue, searchValue], (err, results) => {
    if (err) {
      console.error("Database error during search:", err);
      return res.status(500).json({ error: "Internal server error" });
    }

    if (results.length === 0) {
      return res.status(404).json({ message: "No results found" });
    }

    // จัดกลุ่มข้อมูลโดย StockSymbol
    const groupedResults = results.reduce((acc, stock) => {
      const existingStock = acc.find((item) => item.StockSymbol === stock.StockSymbol);

      if (existingStock) {
        // ถ้ามีอยู่แล้ว เพิ่มข้อมูล ClosePrice เข้าไปในรายการราคา
        existingStock.prices.push({
          date: stock.Date,
          close_price: stock.ClosePrice,
        });
      } else {
        // ถ้ายังไม่มี ให้เพิ่ม StockSymbol และรายละเอียดหุ้น
        acc.push({
          StockSymbol: stock.StockSymbol,
          Market: stock.Market,
          MarketCap: stock.MarketCap,
          CompanyName: stock.CompanyName,
          prices: stock.Date
            ? [
                {
                  date: stock.Date,
                  close_price: stock.ClosePrice,
                },
              ]
            : [], // ถ้าไม่มีราคาหุ้น ให้เป็น array ว่าง
        });
      }

      return acc;
    }, []);

    res.json({ results: groupedResults });
  });
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
      return res.status(400).json({ error: "Invalid birthday format (YYYY-MM-DD expected)" });
    }

    // แปลงรูปแบบวันเกิดให้เป็น YYYY-MM-DD
    birthday = formatDateForSQL(birthday);

    // คำนวณอายุจากวันเกิด
    const age = calculateAge(birthday);

    // เช็คว่า Username ถูกใช้โดยผู้ใช้อื่นหรือไม่
    const checkUsernameSql = `SELECT UserID FROM User WHERE Username = ? AND UserID != ?`;

    pool.query(checkUsernameSql, [username, userId], (checkError, checkResults) => {
      if (checkError) {
        console.error("Error checking username:", checkError);
        return res.status(500).json({ error: "Database error while checking username" });
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
          return res.status(500).json({ error: "Database error while updating user profile" });
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
            profileImage: profileImage || "No image uploaded"
          }
        });
      });
    });
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
      return res.status(500).json({ error: "Database error while retrieving user profile" });
    }

    if (results.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const user = results[0];

    // ✅ ตรวจสอบค่า birthday และ profileImage ก่อนส่งออกไป
    const age = user.Birthday ? calculateAge(user.Birthday) : "N/A";
    const profileImage = user.ProfileImageURL ? user.ProfileImageURL : "/uploads/default.png";

    res.json({
      userId: user.UserID,
      username: user.Username,
      email: user.Email,
      gender: user.Gender,
      birthday: user.Birthday,
      age: age,
      profileImage: profileImage
    });
  });
});

// Helper function: แปลงวันเกิดเป็น YYYY-MM-DD
function formatDateForSQL(dateString) {
  const dateObj = new Date(dateString);
  const year = dateObj.getFullYear();
  const month = String(dateObj.getMonth() + 1).padStart(2, '0'); // Ensure 2 digits
  const day = String(dateObj.getDate()).padStart(2, '0'); // Ensure 2 digits
  return `${year}-${month}-${day}`;
}

// Helper function: คำนวณอายุจากวันเกิด
function calculateAge(birthday) {
  const birthDate = new Date(birthday);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--; // ลดอายุลง 1 ถ้ายังไม่ถึงวันเกิดปีนี้
  }
  return age;
}

// ----Noti---- //

app.get("/api/news-notifications", verifyToken, (req, res) => {
  const today = new Date().toISOString().split("T")[0]; // ดึงวันที่ปัจจุบัน (YYYY-MM-DD)
  
  const fetchNewsNotificationsSql = `
    SELECT 
      n.Title, 
      n.PublishedDate
    FROM News n
    WHERE DATE(n.PublishedDate) = ?
    ORDER BY n.PublishedDate DESC;
  `;

  pool.query(fetchNewsNotificationsSql, [today], (error, results) => {
    if (error) {
      console.error("Database error during fetching news notifications:", error);
      return res.status(500).json({ error: "Error fetching news notifications" });
    }

    res.json({ 
      message: "ข่าวสารประจำวันที่", 
      date: today, 
      news: results 
    });
  });
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
  const checkFollowSql = "SELECT * FROM FollowedStocks WHERE UserID = ? AND StockSymbol = ?";
  pool.query(checkFollowSql, [user_id, stock_symbol], (err, results) => {
    if (err) {
      console.error("Database error during checking followed stock:", err);
      return res.status(500).json({ error: "Database error during checking followed stock" });
    }

    if (results.length > 0) {
      return res.status(400).json({ error: "You are already following this stock" });
    }

    // เพิ่มข้อมูลหุ้นที่ติดตามลงในตาราง FollowedStocks
    const followStockSql = "INSERT INTO FollowedStocks (UserID, StockSymbol, FollowDate) VALUES (?, ?, NOW())";
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
  const deleteFollowedStockSql = "DELETE FROM FollowedStocks WHERE UserID = ? AND StockSymbol = ?";
  pool.query(deleteFollowedStockSql, [user_id, stock_symbol], (err, results) => {
    if (err) {
      console.error("Database error during unfollowing stock:", err);
      return res.status(500).json({ error: "Error unfollowing stock" });
    }

    if (results.affectedRows === 0) {
      return res.status(404).json({ message: "Stock not found in followed list or you are not authorized to remove" });
    }

    res.json({ message: "Stock unfollowed successfully" });
  });
});

// API สำหรับดึงรายการหุ้นที่ผู้ใช้ติดตาม
app.get("/api/favorites", verifyToken, (req, res) => {
  const userId = req.userId; // ดึง userId จาก Token ที่ผ่านการตรวจสอบแล้ว

  // ดึงรายการหุ้นที่ผู้ใช้ติดตามจากฐานข้อมูล
  const fetchFavoritesSql = `
    SELECT 
      fs.StockSymbol, 
      s.CompanyName,
      sd.ClosePrice AS LastPrice,
      sd.Date AS LastUpdated
    FROM FollowedStocks fs
    JOIN Stock s ON fs.StockSymbol = s.StockSymbol
    LEFT JOIN StockDetail sd ON fs.StockSymbol = sd.StockSymbol
    WHERE fs.UserID = ?
    ORDER BY sd.Date DESC;
  `;

  pool.query(fetchFavoritesSql, [userId], (err, results) => {
    if (err) {
      console.error("Database error during fetching favorites:", err);
      return res.status(500).json({ error: "Error fetching favorite stocks" });
    }

    if (results.length === 0) {
      return res.status(404).json({ message: "No followed stocks found" });
    }

    res.json(results);
  });
});

//Top 10 Stock
app.get("/api/top-10-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุดที่มีข้อมูล
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res.status(500).json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0].LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่หุ้นที่มีการเปลี่ยนแปลงสูงสุด 10 อันดับ พร้อมราคาปิด
      const query = `
        SELECT s.StockSymbol, sd.\`Change (%)\` AS ChangePercentage, sd.ClosePrice
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ?
        ORDER BY sd.\`Change (%)\` DESC
        LIMIT 10;
      `;

      pool.query(query, [latestDate], (err, results) => {
        if (err) {
          console.error("Database error fetching top 10 stocks:", err);
          return res.status(500).json({ error: "Database error fetching top 10 stocks" });
        }

        res.json({
          date: latestDate,
          topStocks: results.map(stock => ({
            StockSymbol: stock.StockSymbol,
            ChangePercentage: stock.ChangePercentage,
            ClosePrice: stock.ClosePrice
          }))
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Top 3 Trending Stocks
app.get("/api/trending-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุดที่มีข้อมูล
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res.status(500).json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0].LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // คิวรี่หุ้นที่มีการเปลี่ยนแปลงสูงสุด 3 อันดับแรก พร้อมดึงวันที่
      const query = `
        SELECT sd.Date, s.StockSymbol, s.CompanyName, s.Market, sd.\`Change (%)\` AS ChangePercentage, sd.ClosePrice
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ?
        ORDER BY sd.\`Change (%)\` DESC
        LIMIT 3;
      `;

      pool.query(query, [latestDate], (err, results) => {
        if (err) {
          console.error("Database error fetching trending stocks:", err);
          return res.status(500).json({ error: "Database error fetching trending stocks" });
        }

        res.json({
          date: latestDate,
          trendingStocks: results.map(stock => ({
            Date: stock.Date,  // ✅ ดึงวันที่ที่ใช้จริง
            StockSymbol: stock.StockSymbol,
            CompanyName: stock.CompanyName,
            Market: stock.Market,
            ChangePercentage: stock.ChangePercentage,
            ClosePrice: stock.ClosePrice
          }))
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---- News ---- //

app.get("/api/latest-news", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุด
    const latestDateQuery = "SELECT MAX(PublishedDate) AS LatestDate FROM News";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res.status(500).json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0].LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No news data available" });
      }

      // ดึงข่าวล่าสุดจากตาราง News
      const newsQuery = `
        SELECT Title, Sentiment, PublishedDate 
        FROM News 
        WHERE PublishedDate
        ORDER BY PublishedDate DESC
        LIMIT 10;
      `;

      pool.query(newsQuery, [latestDate], (err, results) => {
        if (err) {
          console.error("Database error fetching latest news:", err);
          return res.status(500).json({ error: "Database error fetching latest news" });
        }

        res.json({
          date: latestDate,
          news: results.map(news => ({
            Title: news.Title,
            Sentiment: news.Sentiment,
            PublishedDate: news.PublishedDate
          }))
        });
      });
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
      SELECT NewsID, Title, Sentiment, Source, PublishedDate, ConfidenceScore, Content, URL
      FROM News
      WHERE NewsID = ?
      LIMIT 1;
    `;

    pool.query(newsDetailQuery, [id], (err, results) => {
      if (err) {
        console.error("Database error fetching news detail:", err);
        return res.status(500).json({ error: "Database error fetching news detail" });
      }

      if (results.length === 0) {
        return res.status(404).json({ error: "News not found" });
      }

      const news = results[0];

      // แปลง ConfidenceScore เป็นเปอร์เซ็นต์
      const confidencePercentage = `${(news.ConfidenceScore * 100).toFixed(0)}%`;

      res.json({
        NewsID: news.NewsID,
        Title: news.Title,
        Sentiment: news.Sentiment,
        Source: news.Source,
        PublishedDate: news.PublishedDate,
        ConfidenceScore: confidencePercentage, // แสดงค่าเป็นเปอร์เซ็นต์
        Content: news.Content,
        URL: news.URL
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});







// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
  