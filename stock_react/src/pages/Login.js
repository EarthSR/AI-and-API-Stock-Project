import React, { useState } from "react";
import axios from "axios";
import styled from "styled-components";
import { motion } from "framer-motion";

const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: linear-gradient(140deg, rgb(26, 26, 26), #334756);
`;

const LoginBox = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
  text-align: center;
  width: 100%;
  max-width: 350px;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const Title = styled.h2`
  color: white;
  font-size: 22px;
  font-weight: bold;
  margin-bottom: 20px;
`;

const Input = styled.input`
  width: 92%;
  padding: 12px;
  margin-top: 12px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  outline: none;
  background: transparent;
  color: white;
  font-size: 16px;
  transition: all 0.3s;

  &::placeholder {
    color: rgba(255, 255, 255, 0.3);
  }

  &:focus {
    border: 1px solid rgb(190, 73, 0);
    background: rgba(255, 255, 255, 0.1);
  }
`;

const Button = styled(motion.button)`
  width: 100%;
  padding: 12px;
  margin-top: 20px;
  background: #f0a500;
  color: #1a1a1d;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: bold;
  transition: all 0.3s;

  &:hover {
    background: rgb(192, 131, 0);
  }

  &:active {
    transform: scale(0.96);
  }
`;

const FooterText = styled.p`
  color: rgba(255, 255, 255, 0.6);
  font-size: 14px;
  margin-top: 15px;
`;

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post("http://localhost:3000/api/admin/login", {
        email,
        password,
      });

      if (response.data.token) {
        // หากล็อกอินสำเร็จ เก็บ Token ใน localStorage
        localStorage.setItem("authToken", response.data.token);
        window.location.href = "/dashboard"; // เปลี่ยนหน้าไปที่ Dashboard
      }
    } catch (err) {
      setError(err.response?.data?.error || "❌ เกิดข้อผิดพลาด");
    }
  };

  return (
    <Container>
      <LoginBox
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Title>Welcome Admin</Title>
        <form onSubmit={handleLogin}>
          <Input
            type="Email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <Input
            type="Password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <p style={{ color: "red" }}>{error}</p>}
          <Button whileTap={{ scale: 0.95 }}>Login</Button>
        </form>
        <FooterText>For Admin Only!!</FooterText>
      </LoginBox>
    </Container>
  );
}

export default Login;
