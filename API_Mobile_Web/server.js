const express = require("express");
const bodyParser = require("body-parser");
const mysql = require("mysql2");
const mysqlpromise = require('mysql2/promise');
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
const serviceAccount = require("./config/trademine-a3921-firebase-adminsdk-fbsvc-ff0de5bd4d.json");




admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors()); // Enable CORS
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));


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

    req.userId = decoded.id;  // ✅ เก็บ userId
    req.role = decoded.role;  // ✅ เก็บ role ไว้ใช้ต่อใน verifyAdmin
    next();
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

// User-Register-Email
app.post("/api/register/email", async (req, res) => {
  try {
      const { email, role = "user" } = req.body; // ถ้า Role ไม่ได้ส่งมา ให้ Default เป็น "user"

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
                  return res.status(400).json({ error: "Email Already use" });
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
              `INSERT INTO User (Email, Username, Password, Role, Status) 
              VALUES (?, '', '', ?, 'pending') 
              ON DUPLICATE KEY UPDATE Status = 'pending', Role = ?`, // ✅ เพิ่ม Role
              [email, role, role], // ✅ กำหนดค่า Role
              (err) => {
                  if (err) {
                      console.error("Database error during User insertion or update:", err);
                      return res.status(500).json({ error: "Database error during User insertion or update" });
                  }

                  // **ดึง UserID จาก Email**
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

                          // **แทรก OTP โดยใช้ UserID แทน Email**
                          pool.query(
                              `INSERT INTO OTP (OTP_Code, Created_At, Expires_At, UserID) 
                              VALUES (?, ?, ?, ?) 
                              ON DUPLICATE KEY UPDATE OTP_Code = ?, Created_At = ?, Expires_At = ?`,
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
          return res.status(400).json({ error: "Invalid OTP" });
        }

        // ตรวจสอบว่า OTP ยังไม่หมดอายุ
        const { Expires_At } = otpResults[0];
        if (new Date() > new Date(Expires_At)) {
          return res.status(400).json({ error: "Expired OTP" });
        }

        // ถ้า OTP ถูกต้องและไม่หมดอายุ
        res.status(200).json({ message: "OTP is correct, you can set a password." });
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
      return res.status(400).json({ error: "Password is required" });
    }

    const hash = await bcrypt.hash(password, 10);

    // ดึง UserID ก่อนอัปเดตรหัสผ่าน
    pool.query("SELECT UserID FROM User WHERE Email = ?", [email], (err, results) => {
      if (err) {
        console.error("Error fetching UserID:", err);
        return res.status(500).json({ error: "Database error fetching UserID" });
      }

      if (results.length === 0) {
        return res.status(404).json({ error: "No account found with this Email" });
      }

      const userId = results[0].UserID;

      // อัปเดตรหัสผ่านในตาราง User แต่ยังไม่เปลี่ยนเป็น 'active'
      pool.query(
        "UPDATE User SET Password = ?, Status = 'active' WHERE UserID = ?",
        [hash, userId],
        (err, results) => {
          if (err) {
            console.error("Database error during User update:", err);
            return res.status(500).json({ error: "Database error during User update" });
          }

          if (results.affectedRows === 0) {
            return res.status(404).json({ error: "Unable to update password" });
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
              message: "Password has been set successfully. Please complete your profile.",
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
      return res.status(400).json({ error: "Please Enter You Email" });
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

// Login
app.post("/api/login", async (req, res) => {
  try {
    const { email, password, googleId } = req.body;
    const ipAddress = req.headers["x-forwarded-for"] || req.connection.remoteAddress;

    if (!email) {
      return res.status(400).json({ message: "กรุณากรอกอีเมล" });
    }

    // ค้นหาผู้ใช้ในฐานข้อมูล
    const [rows] = await pool.promise().query("SELECT * FROM User WHERE Email = ?", [email]);

    if (rows.length === 0) {
      return res.status(404).json({ message: "ไม่พบบัญชีนี้" });
    }

    const user = rows[0];

    // เช็คสถานะบัญชี
    if (user.Status !== "active") {
      return res.status(403).json({ message: "บัญชีถูกระงับการใช้งาน" });
    }

    // ถ้าใช้ Google Login
    if (googleId) {
      if (user.GoogleID && user.GoogleID !== googleId) {
        return res.status(400).json({ message: "บัญชีนี้ถูกลงทะเบียนด้วยอีเมลแล้ว โปรดเข้าสู่ระบบด้วยอีเมลและรหัสผ่าน" });
      }

      if (!user.GoogleID) {
        // อัปเดต GoogleID ในบัญชีที่มีอยู่
        await pool.promise().query("UPDATE User SET GoogleID = ? WHERE UserID = ?", [googleId, user.UserID]);
      }

      // ตรวจสอบว่าอีเมลจาก Google ตรงกับฐานข้อมูลหรือไม่
      if (user.Email !== email) {
        return res.status(400).json({ message: "อีเมลนี้ไม่ตรงกับข้อมูลในระบบ" });
      }

      // สร้าง Token และบันทึกการเข้าสู่ระบบ
      const token = jwt.sign({ id: user.UserID, email: user.Email, role: user.Role }, JWT_SECRET);
      await pool.promise().query("UPDATE User SET LastLogin = NOW(), LastLoginIP = ? WHERE UserID = ?", [ipAddress, user.UserID]);

      console.log(`User logged in with Google: ${user.Email}, Role: ${user.Role}`);
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

    // ถ้าไม่ได้ใส่รหัสผ่าน
    if (!password) {
      return res.status(400).json({ message: "กรุณากรอกรหัสผ่าน" });
    }

    // ป้องกันการล็อกอินผิดพลาดบ่อย
    if (user.FailedAttempts >= 5 && user.LastFailedAttempt) {
      const timeSinceLastAttempt = Date.now() - new Date(user.LastFailedAttempt).getTime();
      if (timeSinceLastAttempt < 300000) {
        return res.status(429).json({ message: "คุณล็อกอินผิดพลาดหลายครั้ง โปรดลองอีกครั้งใน 5 นาที" });
      }
    }

    // ตรวจสอบรหัสผ่าน
    const isMatch = await bcrypt.compare(password, user.Password);
    if (!isMatch) {
      await pool.promise().query("UPDATE User SET FailedAttempts = FailedAttempts + 1, LastFailedAttempt = NOW() WHERE UserID = ?", [user.UserID]);
      return res.status(401).json({ message: "อีเมลหรือรหัสผ่านไม่ถูกต้อง" });
    }

    // รีเซ็ต FailedAttempts และอัปเดตการเข้าสู่ระบบ
    await pool.promise().query("UPDATE User SET FailedAttempts = 0, LastLogin = NOW(), LastLoginIP = ? WHERE UserID = ?", [ipAddress, user.UserID]);

    // สร้าง JWT Token
    const token = jwt.sign({ id: user.UserID, role: user.Role }, JWT_SECRET, { expiresIn: "7d" });

    console.log(`User logged in: ${user.Email}, Role: ${user.Role}`);
    res.status(200).json({
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
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});




// Set Profile และ Login อัตโนมัติหลังจากตั้งโปรไฟล์เสร็จ
app.post("/api/set-profile", verifyToken, upload.single('picture'), (req, res) => {
  const { newUsername, birthday, gender } = req.body;
  const userId = req.userId; // รับ UserID จาก token
  const picture = req.file ? `/uploads/${req.file.filename}` : null;

  // ตรวจสอบค่าที่ต้องมี
  if (!newUsername || !picture || !birthday || !gender) {
    return res.status(400).json({ message: "New username, picture, birthday, and gender are required" });
  }

  // ตรวจสอบว่าค่า gender ถูกต้อง
  const validGenders = ["Male", "Female", "Other"];
  if (!validGenders.includes(gender)) {
    return res.status(400).json({ message: "Invalid gender. Please choose 'Male', 'Female', or 'Other'." });
  }

  // แปลงวันที่จาก DD/MM/YYYY เป็น YYYY-MM-DD
  const birthdayParts = birthday.split('/');
  const formattedBirthday = `${birthdayParts[2]}-${birthdayParts[1]}-${birthdayParts[0]}`;

  // อัปเดตโปรไฟล์ของผู้ใช้ และเปลี่ยนสถานะเป็น Active
  const updateProfileQuery = `
    UPDATE User 
    SET Username = ?, ProfileImageURL = ?, Birthday = ?, Gender = ?, Status = 'active' 
    WHERE UserID = ?`;

  pool.query(updateProfileQuery, [newUsername, picture, formattedBirthday, gender, userId], (err) => {
    if (err) {
      console.error("Error updating profile: ", err);
      return res.status(500).json({ message: "Error updating profile" });
    }

    // ดึงข้อมูลผู้ใช้เพื่อนำไปสร้าง Token
    pool.query("SELECT UserID, Email, Username, ProfileImageURL, Gender FROM User WHERE UserID = ?", [userId], (err, userResults) => {
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
          gender: user.Gender
        },
      });
    });
  });
});



// ---- Search ---- //

app.get("/api/search", (req, res) => {
  const { query } = req.query;

  if (!query) {
    return res.status(400).json({ error: "Search query is required" });
  }

  const searchValue = `%${query.trim().toLowerCase()}%`;

  const searchSql = `
    SELECT 
        s.StockSymbol, 
        s.Market, 
        s.CompanyName, 
        sd.StockDetailID,  -- ดึงเฉพาะวันล่าสุด
        sd.Date, 
        sd.ClosePrice,
        sd.ChangePercen
    FROM Stock s
    INNER JOIN StockDetail sd 
        ON s.StockSymbol = sd.StockSymbol
    INNER JOIN (
        SELECT StockSymbol, MAX(Date) AS LatestDate
        FROM StockDetail
        GROUP BY StockSymbol
    ) latest ON sd.StockSymbol = latest.StockSymbol AND sd.Date = latest.LatestDate
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

    const groupedResults = results.map(stock => ({
      StockSymbol: stock.StockSymbol,
      Market: stock.Market,
      CompanyName: stock.CompanyName,
      StockDetailID: stock.StockDetailID,
      LatestDate: stock.Date,
      ClosePrice: stock.ClosePrice,
      ChangePercen: stock.ChangePercen  
    }));

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


app.post("/api/update-fcm-token", verifyToken, async (req, res) => {
  const { fcm_token } = req.body;

  if (!fcm_token) {
    return res.status(400).json({ error: "fcm_token ห้ามว่าง" });
  }

  try {
    const userId = req.userId;

    const [rows] = await pool_notification.query(
      "SELECT fcm_token FROM user WHERE UserID = ?",
      [userId]
    );

    if (rows.length === 0) {
      return res.status(404).json({ error: "ไม่พบผู้ใช้งาน" });
    }

    const currentToken = rows[0].fcm_token;

    // ถ้า fcm_token ใหม่ไม่เหมือนเดิม หรือเดิมเป็นค่าว่าง ให้ update
    if (!currentToken || currentToken.trim() === "" || currentToken !== fcm_token) {
      await pool_notification.query(
        "UPDATE user SET fcm_token = ? WHERE UserID = ?",
        [fcm_token, userId]
      );
      return res.json({ message: "อัปเดต fcm_token เรียบร้อย", fcm_token });
    }

    // ถ้าเหมือนกัน ไม่ต้อง update
    return res.json({
      message: "fcm_token เหมือนเดิม ไม่ต้องอัปเดต",
      fcm_token: currentToken,
    });

  } catch (err) {
    console.error("❌ Error updating fcm_token:", err);
    res.status(500).json({ error: "เกิดข้อผิดพลาดในระบบ" });
  }
});



app.get("/api/news-notifications", verifyToken, async (req, res) => {
  try {
    // 1. ดึงข่าวล่าสุด
    const [newsResults] = await pool_notification.query(`
      SELECT NewsID, Title, PublishedDate
      FROM News
      ORDER BY PublishedDate DESC
      LIMIT 1;
    `);

    if (newsResults.length === 0) {
      return res.json({ message: "ยังไม่มีข่าวในฐานข้อมูล" });
    }

    const latestNews = newsResults[0]; // ข่าวล่าสุด

    // 2. ดึง fcm_token และ UserID จากตาราง user (สมมติชื่อคอลัมน์ UserID, fcm_token)
    const [userResults] = await pool_notification.query(`
      SELECT UserID, fcm_token
      FROM user
      WHERE fcm_token IS NOT NULL AND fcm_token != ''
    `);

    if (userResults.length === 0) {
      return res.json({ message: "ยังไม่มีผู้ใช้ที่มี fcm_token" });
    }

    // 3. บันทึกลง table notification
    for (const user of userResults) {
      await pool_notification.query(
        `
        INSERT INTO notification (Message, Date, NewsID, UserID)
        VALUES (?, NOW(), ?, ?)
      `,
        [latestNews.Title, latestNews.NewsID, user.UserID]
      );
    }

    // 4. เตรียมข้อความส่ง FCM
    const messages = userResults.map((user) => ({
      notification: {
        title: "📰 ข่าวล่าสุด",
        body: latestNews.Title,
      },
      token: user.fcm_token,
    }));

    // 5. ส่ง Notification
    if (typeof admin.messaging().sendAll === "function") {
      const response = await admin.messaging().sendAll(messages);
      console.log("✅ Notifications sent:", response.successCount, "successes");
      res.json({
        message: "📤 ส่ง Push notification สำเร็จ",
        successCount: response.successCount,
        totalUsers: userResults.length,
        news: latestNews,
      });
    } else {
      const messaging = admin.messaging();
      let successCount = 0;
      for (const msg of messages) {
        try {
          await messaging.send(msg);
          successCount++;
        } catch (error) {
          console.error("Error sending notification:", error);
        }
      }
      console.log("✅ Notifications sent (fallback):", successCount);
      res.json({
        message: "📤 ส่ง Push notification สำเร็จ (fallback)",
        successCount,
        totalUsers: userResults.length,
        news: latestNews,
      });
    }
  } catch (err) {
    console.error("❌ Error pushing notifications:", err);
    res.status(500).json({ error: "เกิดข้อผิดพลาดขณะดึงข่าวหรือส่ง noti" });
  }
});


// API สำหรับดึงข่าว notification ล่าสุดที่ถูกบันทึก (ของทุก user หรือ กรอง userId ก็ได้)
app.get("/api/latest-notification", verifyToken, async (req, res) => {
  try {
    const userId = req.userId;

    const [results] = await pool_notification.query(`
      SELECT n.NotificationID, n.Message, n.Date, n.NewsID, n.UserID, nw.Title AS NewsTitle, nw.PublishedDate
      FROM notification n
      LEFT JOIN News nw ON n.NewsID = nw.NewsID
      WHERE n.UserID = ?
      ORDER BY n.Date DESC
      LIMIT 10;
    `, [userId]);

    if (results.length === 0) {
      return res.json({ message: "ไม่มีการแจ้งเตือนล่าสุด" });
    }

    res.json({
      notifications: results
    });
  } catch (error) {
    console.error("Error fetching latest notifications:", error);
    res.status(500).json({ error: "เกิดข้อผิดพลาดขณะดึงข้อมูลการแจ้งเตือน" });
  }
});






// // ดึงข้อมูลข่าวและ push noti
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
//     const [newsResults] = await pool.promise().query(fetchNewsNotificationsSql, [today]);

//     // ถ้ามีข่าวใหม่
//     if (newsResults.length > 0) {
//       const latestNews = newsResults[0]; // ข่าวล่าสุด

//       // ดึงรายชื่อผู้ใช้พร้อม FCM Token
//       userResults = "fLmzIwKYS2SuSkidLdjGjs:APA91bFnyXm3-myy4U3Eg1yjwR4ahvtmgHdwLHP4WD-e0StfE4ws6A6oP-cn0HkqW_8YN7mwxpCi4-aScGF_kdjI2chdhQmYxkvkpWCfMSVmt1hCz6Vzf8Q"
//       ฝฝconst [userResults] = await pool.promise().query("SELECT fcm_token FROM users WHERE fcm_token IS NOT NULL");

//       // สร้าง message สำหรับแต่ละ user
//       const messages = userResults.map(user => ({
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

// API สำหรับดึงรายการหุ้นที่ผู้ใช้ติดตาม พร้อมราคาวันล่าสุด และกราฟ 5 วัน
app.get("/api/favorites", verifyToken, (req, res) => {
  const userId = req.userId;

  // ✅ ดึงหุ้นที่ผู้ใช้ติดตาม พร้อมชื่อบริษัทและ FollowDate เรียงตาม FollowDate จากใหม่ไปเก่า
  const fetchFavoritesSql = `
    SELECT fs.FollowID, fs.StockSymbol, fs.FollowDate, s.CompanyName ,s.Market
    FROM FollowedStocks fs
    JOIN Stock s ON fs.StockSymbol = s.StockSymbol
    WHERE fs.UserID = ?
    ORDER BY fs.FollowDate DESC;
  `;

  pool.query(fetchFavoritesSql, [userId], (err, stockResults) => {
    if (err) {
      console.error("Database error during fetching favorites:", err);
      return res.status(500).json({ error: "Error fetching favorite stocks" });
    }

    if (stockResults.length === 0) {
      return res.status(404).json({ message: "No followed stocks found" });
    }

    const stockSymbols = stockResults.map(stock => stock.StockSymbol);

    const fetchStockDetailsSql = `
      SELECT StockSymbol, ClosePrice, Changepercen AS ChangePercentage, Date
      FROM StockDetail
      WHERE StockSymbol IN (?) 
      ORDER BY Date DESC;
    `;

    pool.query(fetchStockDetailsSql, [stockSymbols], (err, priceResults) => {
      if (err) {
        console.error("Database error during fetching stock details:", err);
        return res.status(500).json({ error: "Error fetching stock details" });
      }

      const stockDataMap = {};
      stockResults.forEach(stock => {
        stockDataMap[stock.StockSymbol] = {
          FollowID: stock.FollowID,
          StockSymbol: stock.StockSymbol,
          CompanyName: stock.CompanyName,
          FollowDate: stock.FollowDate,
          Market: stock.Market,
          LastPrice: null,
          LastChange: null,
          HistoricalPrices: []
        };
      });

      priceResults.forEach(price => {
        if (!stockDataMap[price.StockSymbol].LastPrice) {
          stockDataMap[price.StockSymbol].LastPrice = price.ClosePrice;
          stockDataMap[price.StockSymbol].LastChange = price.ChangePercentage;
        }

        if (stockDataMap[price.StockSymbol].HistoricalPrices.length < 5) {
          stockDataMap[price.StockSymbol].HistoricalPrices.push({
            Date: price.Date,
            ClosePrice: price.ClosePrice
          });
        }
      });

      res.json(Object.values(stockDataMap));
    });
  });
});


// API สำหรับดึงหุ้นที่มีการเปลี่ยนแปลงสูงสุด 10 อันดับ พร้อมราคาปิด และ ID
app.get("/api/top-10-stocks", async (req, res) => {
  try {
    // ดึงวันที่ล่าสุด
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res.status(500).json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // ดึง 10 หุ้นที่เปอร์เซ็นต์ขึ้นสูงสุดในวันล่าสุด
      const query = `
        SELECT 
          sd.StockDetailID, 
          s.StockSymbol, 
          sd.Changepercen AS ChangePercentage, 
          sd.ClosePrice
        FROM StockDetail sd
        JOIN Stock s ON sd.StockSymbol = s.StockSymbol
        WHERE sd.Date = ?
        ORDER BY sd.Changepercen DESC
        LIMIT 10
      `;

      pool.query(query, [latestDate], (err, results) => {
        if (err) {
          console.error("Database error fetching top 10 stocks:", err);
          return res.status(500).json({ error: "Database error fetching top 10 stocks" });
        }

        res.json({
          date: latestDate,
          topStocks: results.map(stock => ({
            StockDetailID: stock.StockDetailID,
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


// API สำหรับดึง 3 หุ้นที่มีการเปลี่ยนแปลงสูงสุด พร้อมข้อมูลย้อนหลัง 5 วัน
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
          return res.status(500).json({ error: "Database error fetching trending stocks" });
        }

        const stockSymbols = trendingStocks.map(stock => stock.StockSymbol);

        if (stockSymbols.length === 0) {
          return res.status(404).json({ error: "No trending stocks found" });
        }

        // ดึงข้อมูลย้อนหลัง 5 วัน (นับจากวันล่าสุด)
        const historyQuery = `
          SELECT 
            StockSymbol, 
            Date, 
            ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY Date DESC
          LIMIT ?;
        `;

        pool.query(historyQuery, [stockSymbols, stockSymbols.length * 5], (err, historyData) => {
          if (err) {
            console.error("Database error fetching historical data:", err);
            return res.status(500).json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyData.forEach(entry => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice
              });
            }
          });

          // แปลงข้อมูลให้ตรงกับโครงสร้าง JSON ที่ต้องการ
          const response = {
            date: latestDate,
            trendingStocks: trendingStocks.map(stock => {
              const priceChangePercentage = stock.PredictionClose
                ? ((stock.PredictionClose - stock.ClosePrice) / stock.ClosePrice) * 100
                : null;

              let stockType = stock.Market === "America" ? "US Stock" : "TH Stock";

              return {
                StockDetailID: stock.StockDetailID, // ✅ เพิ่ม ID สำหรับอ้างอิง
                Date: stock.Date,
                StockSymbol: stock.StockSymbol,
                ChangePercentage: stock.ChangePercentage,
                ClosePrice: stock.ClosePrice,
                PredictionClose: stock.PredictionClose,
                PricePredictionChange: priceChangePercentage ? priceChangePercentage.toFixed(2) + "%" : "N/A",
                Type: stockType,
                HistoricalPrices: historyMap[stock.StockSymbol] || []
              };
            })
          };

          res.json(response);
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
    const limit = parseInt(req.query.limit) || 20;
    const offset = parseInt(req.query.offset) || 0;
    const sourceInput = req.query.source;
    const sentimentInput = req.query.sentiment;
    const sortOrder = req.query.sort?.toUpperCase() === "ASC" ? "ASC" : "DESC"; // default DESC

    // แปลง source ที่รับมาจาก Flutter ให้ตรงกับในฐานข้อมูล
    let sourceDbValue = null;
    if (sourceInput === "TH News") {
      sourceDbValue = "bangkokpost";
    } else if (sourceInput === "US News") {
      sourceDbValue = "investing";
    }

    let query = `
      SELECT 
        NewsID, 
        Title,
        Source,
        Sentiment, 
        DATE_FORMAT(PublishedDate, '%Y-%m-%d') as PublishedDate,
        Img
      FROM News
    `;

    const conditions = [];
    const params = [];

    if (sourceDbValue) {
      conditions.push("Source = ?");
      params.push(sourceDbValue);
    }

    if (sentimentInput && ['Positive', 'Negative', 'Neutral'].includes(sentimentInput)) {
      conditions.push("Sentiment = ?");
      params.push(sentimentInput);
    }

    if (conditions.length > 0) {
      query += " WHERE " + conditions.join(" AND ");
    }

    query += ` ORDER BY PublishedDate ${sortOrder} LIMIT ? OFFSET ?`;
    params.push(limit, offset);

    pool.query(query, params, (err, results) => {
      if (err) {
        console.error("DB Error:", err);
        return res.status(500).json({ error: "Database error" });
      }

      res.json({ news: results });
    });
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "Internal error" });
  }
});


app.get("/api/news-by-source", async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    const offset = parseInt(req.query.offset) || 0;
    const region = (req.query.region || "TH").toUpperCase();

    // แปลงรหัสประเทศ → ชื่อแหล่งข่าวในฐานข้อมูล
    let sourceName;
    if (region === "TH News") {
      sourceName = "BangkokPost";
    } else if (region === "US News") {
      sourceName = "Investing";
    } else {
      return res.status(400).json({ error: "Invalid region" });
    }

    const query = `
      SELECT NewsID, Title, Source, Sentiment, PublishedDate, Img
      FROM News
      WHERE Source = ?
      ORDER BY PublishedDate DESC
      LIMIT ? OFFSET ?;
    `;

    pool.query(query, [sourceName, limit, offset], (err, results) => {
      if (err) {
        console.error("DB Error:", err);
        return res.status(500).json({ error: "Database error" });
      }

      res.json({ news: results });
    });
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "Internal error" });
  }
});

app.get("/api/recommentnews-stockdetail", async (req, res) => {
  try {
    const stockSymbol = req.query.symbol;

    if (!stockSymbol) {
      return res.status(400).json({ error: "Missing StockSymbol in query parameters" });
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
        return res.status(500).json({ error: "Database error fetching news detail" });
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
      return res.status(400).json({ error: "Missing StockSymbol in query parameters" });
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
        return res.status(500).json({ error: "Database error fetching news detail" });
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
      ConfidenceScore: confidencePercentage, 
      Content: news.Content,
      ImageURL: news.Img, 
      URL: news.URL
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
    const response = await fetch("https://api.exchangerate-api.com/v4/latest/USD");
    const data = await response.json();
    return data.rates.THB || 1; // ถ้าค่า THB ไม่มีให้คืนค่า 1 (ไม่แปลง)
  } catch (error) {
    console.error("Error fetching exchange rate:", error);
    return 1; // ถ้าดึงค่าไม่ได้ให้ใช้ค่า 1 (ป้องกัน error)
  }
}

// API ดึงรายละเอียดหุ้น
app.get("/api/stock-detail/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = "5D" } = req.query;

    // กำหนดช่วงเวลาที่รองรับ
    const historyLimits = { 
      "1D": 1, 
      "5D": 5, 
      "1M": 22,    // ประมาณ 22 วันทำการใน 1 เดือน
      "3M": 66,    // 3 เดือน 
      "6M": 132,   // 6 เดือน
      "1Y": 264,   // 1 ปี (ประมาณ 252 วันทำการ)
      "ALL": null  // null หมายถึงดึงข้อมูลทั้งหมด
    };

    if (!historyLimits.hasOwnProperty(timeframe)) {
      return res.status(400).json({ error: "Invalid timeframe. Choose from 1D, 5D, 1M, 3M, 6M, 1Y, ALL." });
    }

    // ดึงวันที่ล่าสุดที่มีข้อมูล
    const latestDateQuery = "SELECT MAX(Date) AS LatestDate FROM StockDetail";
    pool.query(latestDateQuery, async (dateErr, dateResults) => {
      if (dateErr) {
        console.error("Database error fetching latest date:", dateErr);
        return res.status(500).json({ error: "Database error fetching latest date" });
      }

      const latestDate = dateResults[0]?.LatestDate;
      if (!latestDate) {
        return res.status(404).json({ error: "No stock data available" });
      }

      // ดึงข้อมูลหลักของหุ้น
      const stockQuery = `
        SELECT 
          sd.StockDetailID, 
          s.StockSymbol, 
          s.Market, 
          s.CompanyName, 
          s.Sector, 
          s.Industry, 
          s.Description, 
          sd.OpenPrice, 
          sd.HighPrice,
          sd.ClosePrice, 
          sd.MarketCap,
          sd.Changepercen AS ChangePercentage, 
          sd.Volume
        FROM Stock s
        LEFT JOIN StockDetail sd ON s.StockSymbol = sd.StockSymbol AND sd.Date = ?
        WHERE s.StockSymbol = ?;
      `;

      pool.query(stockQuery, [latestDate, symbol], async (err, results) => {
        if (err) {
          console.error("Database error fetching stock details:", err);
          return res.status(500).json({ error: "Database error fetching stock details" });
        }

        if (results.length === 0) {
          return res.status(404).json({ error: "Stock not found" });
        }

        const stock = results[0];
        let stockType = stock.Market === "America" ? "US Stock" : "TH Stock";

        let exchangeRate = 1;
        if (stockType === "US Stock") {
          exchangeRate = await getExchangeRate();
        }

        const closePrice = stock.ClosePrice !== null ? parseFloat(stock.ClosePrice) : 0;
        const closePriceTHB = stockType === "US Stock" ? closePrice * exchangeRate : closePrice;

        let pricePredictionChange = stock.PredictionClose
          ? ((stock.PredictionClose - stock.ClosePrice) / stock.ClosePrice) * 100
          : null;

        const avgVolumeQuery = `
          SELECT AVG(Volume) AS AvgVolume30D 
          FROM StockDetail 
          WHERE StockSymbol = ? 
          ORDER BY Date DESC 
          LIMIT 30;
        `;
        

        pool.query(avgVolumeQuery, [symbol], (volErr, volResults) => {
          if (volErr) {
            console.error("Database error fetching average volume:", volErr);
            return res.status(500).json({ error: "Database error fetching average volume" });
          }

          const avgVolume30D = volResults[0]?.AvgVolume30D ? parseFloat(volResults[0].AvgVolume30D) : 0;
          const formattedAvgVolume30D = avgVolume30D > 0 ? avgVolume30D.toFixed(2) + " Million" : "N/A";

          let historyQuery = `
            SELECT StockSymbol, Date, OpenPrice, HighPrice, LowPrice, ClosePrice
            FROM StockDetail 
            WHERE StockSymbol = ? 
            ORDER BY Date DESC
          `;

          const limit = historyLimits[timeframe];
          if (limit !== null) {
            historyQuery += ` LIMIT ${limit}`;
          }
          // ถ้า timeframe เป็น ALL จะไม่มี LIMIT

          pool.query(historyQuery, [symbol], (histErr, historyResults) => {
            if (histErr) {
              console.error(`Database error fetching historical data:`, histErr);
              return res.status(500).json({ error: "Database error fetching historical data" });
            }

            res.json({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              Type: stockType,
              company : stock.CompanyName,
              ClosePrice: stock.ClosePrice,
              ClosePriceTHB: closePriceTHB.toFixed(2),
              Date: latestDate,
              Change: stock.ChangePercentage,
              PredictionClose: stock.PredictionClose,
              PredictionTrend: stock.PredictionTrend,
              PredictionCloseDate: latestDate,
              PricePredictionChange: pricePredictionChange ? pricePredictionChange.toFixed(2) + "%" : "N/A",
              SelectedTimeframe: timeframe,
              HistoricalPrices: historyResults,
              Overview: {
                High: stock.HighPrice,
                Open: stock.OpenPrice,
                Close: stock.ClosePrice,
                AvgVolume30D: formattedAvgVolume30D,
                Marketcap : stock.MarketCap
              },
              Profile: {
                Market: stock.Market,
                Sector: stock.Sector,
                Industry: stock.Industry,
                Description: stock.Description
              }
            });
          });
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
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
        return res.status(500).json({ error: "Database error fetching latest date" });
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
          return res.status(500).json({ error: "Database error fetching recommended stocks" });
        }

        const stockSymbols = recommendResults.map(stock => stock.StockSymbol);

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
            return res.status(500).json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyResults.forEach(entry => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice
              });
            }
          });

          // ส่ง Response
          res.json({
            date: latestDate,
            recommendedStocks: recommendResults.map(stock => ({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              ClosePrice: stock.ClosePrice,
              Change: stock.ChangePercentage,
              HistoricalPrices: historyMap[stock.StockSymbol] || []
            }))
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
        return res.status(500).json({ error: "Database error fetching latest date" });
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

      pool.query(mostHeldQuery, [latestDate], (mostHeldErr, mostHeldResults) => {
        if (mostHeldErr) {
          console.error("Database error fetching most held stocks:", mostHeldErr);
          return res.status(500).json({ error: "Database error fetching most held stocks" });
        }

        const stockSymbols = mostHeldResults.map(stock => stock.StockSymbol);

        // ดึงข้อมูลกราฟย้อนหลัง 5 วัน
        const historyQuery = `
          SELECT StockSymbol, Date, ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY StockSymbol, Date DESC;
        `;

        pool.query(historyQuery, [stockSymbols], (histErr, historyResults) => {
          if (histErr) {
            console.error("Database error fetching historical data:", histErr);
            return res.status(500).json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyResults.forEach(entry => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice
              });
            }
          });

          // ส่ง Response
          res.json({
            date: latestDate,
            mostHeldStocks: mostHeldResults.map(stock => ({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              Type: "US Stock",
              ClosePrice: stock.ClosePrice,
              Change: stock.ChangePercentage,
              HistoricalPrices: historyMap[stock.StockSymbol] || []
            }))
          });
        });
      });
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
        return res.status(500).json({ error: "Database error fetching latest date" });
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
          console.error("Database error fetching recommended Thai stocks:", recErr);
          return res.status(500).json({ error: "Database error fetching recommended Thai stocks" });
        }

        const stockSymbols = recommendResults.map(stock => stock.StockSymbol);

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
            return res.status(500).json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyResults.forEach(entry => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice
              });
            }
          });

          // ส่ง Response
          res.json({
            date: latestDate,
            recommendedStocks: recommendResults.map(stock => ({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              ClosePrice: stock.ClosePrice,
              Change: stock.ChangePercentage,
              HistoricalPrices: historyMap[stock.StockSymbol] || []
            }))
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
        return res.status(500).json({ error: "Database error fetching latest date" });
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

      pool.query(mostHeldQuery, [latestDate], (mostHeldErr, mostHeldResults) => {
        if (mostHeldErr) {
          console.error("Database error fetching most held Thai stocks:", mostHeldErr);
          return res.status(500).json({ error: "Database error fetching most held Thai stocks" });
        }

        const stockSymbols = mostHeldResults.map(stock => stock.StockSymbol);

        // ดึงข้อมูลกราฟย้อนหลัง 5 วัน
        const historyQuery = `
          SELECT StockSymbol, Date, ClosePrice
          FROM StockDetail
          WHERE StockSymbol IN (?) 
          ORDER BY StockSymbol, Date DESC;
        `;

        pool.query(historyQuery, [stockSymbols], (histErr, historyResults) => {
          if (histErr) {
            console.error("Database error fetching historical data:", histErr);
            return res.status(500).json({ error: "Database error fetching historical data" });
          }

          // จัดกลุ่มข้อมูลย้อนหลังตาม StockSymbol
          const historyMap = {};
          historyResults.forEach(entry => {
            if (!historyMap[entry.StockSymbol]) {
              historyMap[entry.StockSymbol] = [];
            }
            if (historyMap[entry.StockSymbol].length < 5) {
              historyMap[entry.StockSymbol].push({
                Date: entry.Date,
                ClosePrice: entry.ClosePrice
              });
            }
          });

          // ส่ง Response
          res.json({
            date: latestDate,
            mostHeldStocks: mostHeldResults.map(stock => ({
              StockDetailID: stock.StockDetailID,
              StockSymbol: stock.StockSymbol,
              Type: "TH Stock",
              ClosePrice: stock.ClosePrice,
              Change: stock.ChangePercentage,
              HistoricalPrices: historyMap[stock.StockSymbol] || []
            }))
          });
        });
      });
    });
  } catch (error) {
    console.error("Internal server error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


const API_KEY = process.env.FINNHUB_API_KEY; // ใส่ key ใน .env
const cheerio = require('cheerio');
async function getTradingViewPrice(symbol, market = 'thailand', retries = 3) {
  const marketConfig = {
    thailand: { endpoint: 'https://scanner.tradingview.com/thailand/scan', prefixes: ['SET:'] },
    usa: { endpoint: 'https://scanner.tradingview.com/america/scan', prefixes: ['NASDAQ:', 'NYSE:'] }
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
        query: { types: [] }
      },
      columns: ['close', 'description', 'name']
    };

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await axios.post(endpoint, payload, {
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' // เพิ่ม User-Agent
          },
          timeout: 5000 // Timeout 5 วินาที
        });

        console.log(`พยายามครั้งที่ ${attempt} - การตอบสนอง API สำหรับ ${ticker} ใน ${market}:`, response.data);

        const result = response.data?.data?.[0];
        if (!result) {
          throw new Error(`ไม่พบราคาหุ้น ${symbol} (ticker: ${ticker}) ในตลาด ${market}`);
        }

        return {
          symbol: result.d[2],
          name: result.d[1],
          price: result.d[0]
        };
      } catch (error) {
        console.error(`พยายามครั้งที่ ${attempt} ล้มเหลวสำหรับ ${ticker} ใน ${market}:`, error.message);
        lastError = error;
        if (attempt < retries) {
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); // Exponential backoff
        }
      }
    }
  }

  throw new Error(`ไม่สามารถดึงราคาหุ้น ${symbol} ใน ${market} ได้: ${lastError.message}`);
}

app.get('/api/price/:market/:symbol', async (req, res) => {
  const { symbol, market } = req.params;

  try {
    const data = await getTradingViewPrice(symbol, market);
    res.json({
      symbol: data.symbol,
      name: data.name,
      price: data.price,
      market: market.toLowerCase()
    });
  } catch (e) {
    res.status(500).json({ detail: 'ไม่สามารถดึงราคาได้: ' + e.message });
  }
});

// Helper function to get THB to USD exchange rate
async function getThbToUsdRate() {
  try {
    const response = await axios.get(`https://api.exchangerate-api.com/v4/latest/THB`);
    // Return the rate for 1 THB to USD
    return response.data.rates.USD || (1 / 36.5); // Fallback to a rough estimate
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
    const thbToUsdRate = await getThbToUsdRate();

    // 1. ดึงข้อมูล portfolio หลักของผู้ใช้
    const [portfolioRows] = await connection.query(
      "SELECT * FROM papertradeportfolio WHERE UserID = ?",
      [req.userId]
    );

    if (portfolioRows.length === 0) {
      return res.status(404).json({ message: "ไม่พบ Portfolio สำหรับผู้ใช้นี้" });
    }

    const portfolio = portfolioRows[0];

    // 2. ดึงข้อมูลหุ้นทั้งหมดที่ถือครองในพอร์ต พร้อม Market
    const [holdingsRows] = await connection.query(
      `SELECT 
         h.PaperHoldingID, 
         h.StockSymbol, 
         h.Quantity, 
         h.BuyPrice,
         s.Market
       FROM paperportfolioholdings h
       JOIN Stock s ON h.StockSymbol = s.StockSymbol
       WHERE h.PaperPortfolioID = ?`,
      [portfolio.PaperPortfolioID]
    );

    // 3. ดึงราคาปัจจุบันของหุ้นทุกตัวพร้อมกันเพื่อประสิทธิภาพ
    const pricePromises = holdingsRows.map(holding => {
      const apiSymbol = holding.Market === 'Thailand' ? `${holding.StockSymbol.toUpperCase()}.BK` : holding.StockSymbol.toUpperCase();
      const url = `https://finnhub.io/api/v1/quote?symbol=${apiSymbol}&token=${process.env.FINNHUB_API_KEY}`;
      return axios.get(url)
        .then(response => ({ symbol: holding.StockSymbol, price: response.data.c || 0 }))
        .catch(error => {
          console.error(`Could not fetch price for ${holding.StockSymbol}:`, error.message);
          return { symbol: holding.StockSymbol, price: 0, error: true };
        });
    });

    const prices = await Promise.all(pricePromises);
    const priceMap = prices.reduce((map, item) => {
      map[item.symbol] = item.price;
      return map;
    }, {});

    // 4. จัดกลุ่มหุ้นตามสัญลักษณ์ (StockSymbol) และคำนวณค่ารวม
    const groupedHoldings = holdingsRows.reduce((acc, holding) => {
      const symbol = holding.StockSymbol;
      if (!acc[symbol]) {
        acc[symbol] = {
          StockSymbol: symbol,
          Market: holding.Market,
          TotalQuantity: 0,
          TotalCostBasis: 0,
        };
      }
      acc[symbol].TotalQuantity += holding.Quantity;
      acc[symbol].TotalCostBasis += holding.BuyPrice * holding.Quantity;
      return acc;
    }, {});

    // 5. คำนวณมูลค่าและกำไร/ขาดทุนสำหรับแต่ละกลุ่ม และรวมมูลค่าพอร์ตทั้งหมดในสกุลเงิน USD
    let totalHoldingsValueUSD = 0;
    const holdingsWithPL = Object.values(groupedHoldings).map(group => {
      const currentPrice = priceMap[group.StockSymbol] || 0; // ราคาในสกุลเงินเดิม
      const isThaiStock = group.Market === 'Thailand';
      
      // แปลงค่าเงินทั้งหมดเป็น USD เพื่อการแสดงผลที่สอดคล้องกัน
      const costBasisUSD = isThaiStock ? group.TotalCostBasis * thbToUsdRate : group.TotalCostBasis;
      const currentValueUSD = isThaiStock ? (currentPrice * group.TotalQuantity) * thbToUsdRate : (currentPrice * group.TotalQuantity);
      const avgBuyPriceUSD = group.TotalQuantity > 0 ? costBasisUSD / group.TotalQuantity : 0;
      const currentPriceUSD = isThaiStock ? currentPrice * thbToUsdRate : currentPrice;

      const unrealizedPL_USD = currentValueUSD - costBasisUSD;
      const unrealizedPLPercent = costBasisUSD > 0 ? (unrealizedPL_USD / costBasisUSD) * 100 : 0;
      
      totalHoldingsValueUSD += currentValueUSD;

      return {
        StockSymbol: group.StockSymbol,
        Quantity: group.TotalQuantity,
        AvgBuyPriceUSD: avgBuyPriceUSD.toFixed(2),
        CurrentPriceUSD: currentPriceUSD.toFixed(2),
        CurrentValueUSD: currentValueUSD.toFixed(2),
        UnrealizedPL_USD: unrealizedPL_USD.toFixed(2),
        UnrealizedPLPercent: unrealizedPLPercent.toFixed(2) + '%',
        Market: group.Market,
        MarketStatus: getMarketStatus(group.Market)
      };
    });

    // 6. คำนวณมูลค่ารวมของพอร์ตเป็น USD (เงินสดในพอร์ตเป็น USD อยู่แล้ว)
    const balanceUSD = parseFloat(portfolio.Balance); // Balance in DB is already USD
    portfolio.TotalPortfolioValueUSD = balanceUSD + totalHoldingsValueUSD; 
    portfolio.BalanceUSD = balanceUSD.toFixed(2);
    portfolio.holdings = holdingsWithPL;

    res.status(200).json({
      message: "ดึงข้อมูล Portfolio สำเร็จ",
      data: portfolio,
    });
  } catch (error) {
    console.error("Error fetching portfolio:", error);
    res.status(500).json({ error: "Internal server error" });
  } finally {
    if (connection) connection.release();
  }
});

// API สำหรับการซื้อ-ขายหุ้นในพอร์ตจำลอง
app.post("/api/portfolio/trade", verifyToken, async (req, res) => {
  let connection;
  try {
    // 1. ตรวจสอบและแปลงข้อมูลนำเข้า
    let { stockSymbol, quantity, tradeType } = req.body; // 'buy' or 'sell'
    const userId = req.userId;

    if (!stockSymbol || !quantity || !tradeType || !['buy', 'sell'].includes(tradeType)) {
      return res.status(400).json({ error: "กรุณาระบุข้อมูลให้ครบถ้วน: stockSymbol, quantity, และ tradeType ('buy' หรือ 'sell')" });
    }
    const parsedQuantity = parseInt(quantity, 10);
    if (isNaN(parsedQuantity) || parsedQuantity <= 0) {
      return res.status(400).json({ error: "จำนวนหุ้นไม่ถูกต้อง" });
    }
    const normalizedSymbol = stockSymbol.toUpperCase().replace('.BK', '');

    // เริ่ม Transaction
    connection = await pool.promise().getConnection();
    await connection.beginTransaction();

    // 2. ดึงข้อมูลตลาดของหุ้น และตรวจสอบสถานะตลาด
    const [stockInfoRows] = await connection.query("SELECT Market FROM Stock WHERE StockSymbol = ?", [normalizedSymbol]);
    if (stockInfoRows.length === 0) {
      await connection.rollback();
      return res.status(404).json({ error: `ไม่พบข้อมูลหุ้น ${normalizedSymbol}` });
    }
    const market = stockInfoRows[0].Market; // 'Thailand' or 'America'

    const marketStatus = getMarketStatus(market);
    if (marketStatus === 'CLOSED') {
      await connection.rollback();
      return res.status(400).json({ error: `ตลาดหุ้นปิดทำการ ไม่สามารถทำรายการได้` });
    }

    // 3. ดึงราคาหุ้นปัจจุบันด้วยฟังก์ชันใหม่
    let currentPrice;
    try {
      const tradingViewMarket = market === 'Thailand' ? 'thailand' : 'usa';
      const priceData = await getTradingViewPrice(normalizedSymbol, tradingViewMarket);
      currentPrice = priceData.price;
    } catch (e) {
      await connection.rollback();
      console.error("TradingView API error:", e.message);
      return res.status(500).json({ error: `เกิดข้อผิดพลาดในการดึงราคาหุ้น ${normalizedSymbol}` });
    }

    // 4. ดึงข้อมูลพอร์ตและจัดการสกุลเงิน
    const [portfolioRows] = await connection.query("SELECT * FROM papertradeportfolio WHERE UserID = ?", [userId]);
    if (portfolioRows.length === 0) {
      await connection.rollback();
      return res.status(404).json({ message: "ไม่พบพอร์ตการลงทุน กรุณาสร้างพอร์ตก่อน" });
    }
    const portfolio = portfolioRows[0];
    const portfolioId = portfolio.PaperPortfolioID;
    let balanceUSD = parseFloat(portfolio.Balance); // Balance is now treated as USD

    const isThaiStock = market === 'Thailand';
    let totalCostOrValueUSD;

    if (isThaiStock) {
      // Stock price is in THB, convert cost to USD for balance calculation
      const thbToUsdRate = await getThbToUsdRate();
      const totalCostTHB = parsedQuantity * currentPrice;
      totalCostOrValueUSD = totalCostTHB * thbToUsdRate;
    } else {
      // Stock price is already in USD
      totalCostOrValueUSD = parsedQuantity * currentPrice;
    }

    // 5. ประมวลผลการซื้อขาย
    if (tradeType === 'buy') {
      if (balanceUSD < totalCostOrValueUSD) {
        await connection.rollback();
        return res.status(400).json({ error: "ยอดเงินคงเหลือไม่เพียงพอ" });
      }

      await connection.query("UPDATE papertradeportfolio SET Balance = ? WHERE PaperPortfolioID = ?", [balanceUSD - totalCostOrValueUSD, portfolioId]);

      // Insert holding with its native price (THB for Thai, USD for US)
      await connection.query(
        "INSERT INTO paperportfolioholdings (PaperPortfolioID, StockSymbol, Quantity, BuyPrice) VALUES (?, ?, ?, ?)",
        [portfolioId, normalizedSymbol, parsedQuantity, currentPrice]
      );
    } else { // 'sell'
      const [holdingRows] = await connection.query(
        "SELECT * FROM paperportfolioholdings WHERE PaperPortfolioID = ? AND StockSymbol = ? ORDER BY PaperHoldingID ASC",
        [portfolioId, normalizedSymbol]
      );

      const totalHeldQuantity = holdingRows.reduce((sum, row) => sum + row.Quantity, 0);
      if (totalHeldQuantity < parsedQuantity) {
        await connection.rollback();
        return res.status(400).json({ error: "จำนวนหุ้นที่ต้องการขายไม่เพียงพอ" });
      }

      await connection.query("UPDATE papertradeportfolio SET Balance = ? WHERE PaperPortfolioID = ?", [balanceUSD + totalCostOrValueUSD, portfolioId]);

      let quantityToSell = parsedQuantity;
      for (const holding of holdingRows) {
        if (quantityToSell <= 0) break;

        const sellFromThisLot = Math.min(quantityToSell, holding.Quantity);
        quantityToSell -= sellFromThisLot;

        const newLotQuantity = holding.Quantity - sellFromThisLot;
        if (newLotQuantity > 0) {
          await connection.query(
            "UPDATE paperportfolioholdings SET Quantity = ? WHERE PaperHoldingID = ?",
            [newLotQuantity, holding.PaperHoldingID]
          );
        } else {
          await connection.query("DELETE FROM paperportfolioholdings WHERE PaperHoldingID = ?", [holding.PaperHoldingID]);
        }
      }
    }

    // 6. Commit Transaction
    await connection.commit();
    res.status(200).json({
      message: `ทำรายการ ${tradeType === 'buy' ? 'ซื้อ' : 'ขาย'} สำเร็จ`,
      trade: { type: tradeType, symbol: normalizedSymbol, quantity: parsedQuantity, price: currentPrice }
    });

  } catch (error) {
    if (connection) await connection.rollback();
    console.error("Error executing trade:", error);
    res.status(500).json({ error: "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์" });
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



//-----------------------------------------------------------------------------------------------------------------------------------------------//

// Middleware to verify admin role
const verifyAdmin = (req, res, next) => {
  if (req.role !== "admin") {
    return res.status(403).json({ error: "Unauthorized access" });
  }
  next();
};

// API ให้ React ดึง Secure Embed URL ไปใช้
app.get("/get-embed-url", (req, res) => {
  res.json({
    embedUrl:
      "https://app.powerbi.com/view?r=eyJrIjoiOGU0ZjNhMjktYjJiZC00ODA1LWIzM2EtNzNkNDg0NzhhMzVkIiwidCI6IjU3ZDY5NWQ0LWFkODYtNDRkMy05Yzk1LTcxNzZkZWFjZjAzZCIsImMiOjEwfQ%3D%3D",
    embedUrl:
      "https://app.powerbi.com/view?r=eyJrIjoiY2VlNGQ1MTItMTE1Zi00ODkyLThhYjEtMjg4MWRkOTRkZWI2IiwidCI6IjU3ZDY5NWQ0LWFkODYtNDRkMy05Yzk1LTcxNzZkZWFjZjAzZCIsImMiOjEwfQ%3D%3D",
    });
});

//Admin//
app.post("/api/admin/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "กรุณากรอกอีเมลและรหัสผ่าน" });
    }

    // ค้นหาผู้ใช้ที่เป็น Admin และมีสถานะ Active
    const sql = "SELECT * FROM User WHERE Email = ? AND Status = 'active' AND Role = 'admin'";
    pool.query(sql, [email], (err, results) => {
      if (err) {
        console.error("Database error during admin login:", err);
        return res.status(500).json({ error: "เกิดข้อผิดพลาดระหว่างการเข้าสู่ระบบ" });
      }

      if (results.length === 0) {
        return res.status(404).json({ message: "ไม่พบบัญชีแอดมิน หรืออาจถูกระงับ" });
      }

      const user = results[0];

      // ตรวจสอบรหัสผ่าน
      bcrypt.compare(password, user.Password, (err, isMatch) => {
        if (err) {
          console.error("Password comparison error:", err);
          return res.status(500).json({ error: "เกิดข้อผิดพลาดในการตรวจสอบรหัสผ่าน" });
        }

        if (!isMatch) {
          return res.status(401).json({ message: "อีเมลหรือรหัสผ่านไม่ถูกต้อง" });
        }

        // ✅ สร้าง JWT Token (ไม่มี LastLogin / LastLoginIP)
        const token = jwt.sign({ id: user.UserID, role: user.Role }, JWT_SECRET, { expiresIn: "7d" });

        // ✅ ส่งข้อมูล Response
        res.status(200).json({
          message: "เข้าสู่ระบบแอดมินสำเร็จ",
          token,
          user: {
            id: user.UserID,
            email: user.Email,
            username: user.Username,
            profile_image: user.ProfileImageURL,
            role: user.Role
          },
        });
      });
    });
  } catch (error) {
    console.error("Internal error:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});


// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
  