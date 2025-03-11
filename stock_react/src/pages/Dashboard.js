import React, { useState, useEffect } from "react";
import PowerBIEmbed from "./PowerBIEmbed";
import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(140deg, rgb(26, 26, 26), #334756);
  overflow: hidden;
`;

const Header = styled.header`
  width: 100%;
  background: #ff8c00;
  padding: 12px;
  text-align: center;
  color: white;
  font-size: 24px;
  font-weight: bold;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
`;

const ContentContainer = styled.div`
  display: flex;
  flex: 1;
`;

const Sidebar = styled.div`
  width: 250px;
  background: #1e1e1e;
  padding: 20px;
  color: white;
  font-weight: bold;
  box-shadow: 4px 0 10px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
`;

const PowerBIContainer = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 15px;
`;

const FrameContainer = styled.div`
  width: 98%;
  height: 90%;
  background: white;
  padding: 15px;
  border-radius: 12px;
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
`;

const Button = styled.button`
  background-color: ${(props) => (props.active ? "#F0A500" : "#444")};
  color: white;
  padding: 10px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: bold;
  border: none;
  width: 100%;
  transition: background 0.3s;

  &:hover {
    background: #f0a500;
  }
`;

function Dashboard() {
  const navigate = useNavigate();
  const [dashboards, setDashboards] = useState([]);
  const [selectedDashboard, setSelectedDashboard] = useState(null);

  // 📌 **ตรวจสอบ Token ก่อนโหลด Dashboard**
  useEffect(() => {
    const token = localStorage.getItem("authToken");
    if (!token) {
      navigate("/"); // ถ้าไม่มี Token ให้กลับไปที่หน้า Login
    }
  }, [navigate]);

  // 📌 **ดึงข้อมูล Embed URLs จากเซิร์ฟเวอร์**
  useEffect(() => {
    axios
      .get("http://localhost:3000/get-embed-urls")
      .then((response) => {
        setDashboards(response.data.dashboards);
        if (response.data.dashboards.length > 0) {
          setSelectedDashboard(response.data.dashboards[0].embedUrl);
        }
      })
      .catch((error) => {
        console.error("Error fetching dashboards:", error);
      });
  }, []);

  // ✅ **ฟังก์ชัน Logout**
  const handleLogout = () => {
    if (window.confirm("คุณต้องการออกจากระบบใช่หรือไม่?")) {
      localStorage.removeItem("authToken"); // ลบ Token ออกจาก Local Storage
      navigate("/"); // เปลี่ยนหน้าไปที่หน้า Login
    }
  };

  return (
    <DashboardContainer>
      <Header>Stock Market Performance</Header>
      <ContentContainer>
        <Sidebar>
          <h2>📊 Power BI Dashboard</h2>
          {/* 🔹 สร้างปุ่มจาก dashboards ที่ดึงมา */}
          {dashboards.map((dashboard) => (
            <Button
              key={dashboard.id}
              active={selectedDashboard === dashboard.embedUrl}
              onClick={() => setSelectedDashboard(dashboard.embedUrl)}
            >
              {dashboard.name}
            </Button>
          ))}
          {/* 🔹 ปุ่มไปหน้า Manage Users */}
          <Button onClick={() => navigate("/manageuser")}>
            Manage Users
          </Button>
          {/* 🔹 ปุ่ม Logout */}
          <Button color="red" onClick={handleLogout}>
            Logout
          </Button>
        </Sidebar>
        <PowerBIContainer>
          <FrameContainer>
            {selectedDashboard ? (
              <PowerBIEmbed embedUrl={selectedDashboard} />
            ) : (
              <p style={{ textAlign: "center", color: "#fff" }}>Loading Dashboard...</p>
            )}
          </FrameContainer>
        </PowerBIContainer>
      </ContentContainer>
    </DashboardContainer>
  );
}

export default Dashboard;
