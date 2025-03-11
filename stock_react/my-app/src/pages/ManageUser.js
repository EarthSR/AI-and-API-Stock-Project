import React, { useEffect, useState } from "react";
import axios from "axios";
import styled from "styled-components";
import { useNavigate } from "react-router-dom";

const ManageUserContainer = styled.div`
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
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
  margin-right: 20px;
`;

const Button = styled.button`
  background-color: ${(props) => props.color || "#444"};
  color: white;
  padding: 10px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: bold;
  border: none;
  transition: background 0.3s;

  &:hover {
    background: ${(props) => props.hoverColor || "#666"};
  }
`;

const ContentContainer = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  background: white;
  border-radius: 8px;
  overflow: hidden;
`;

const Th = styled.th`
  background: #ff8c00;
  padding: 10px;
  color: white;
  text-align: left;
`;

const Td = styled.td`
  padding: 10px;
  border-bottom: 1px solid #ddd;
`;

function ManageUser() {
  const navigate = useNavigate();
  const [users, setUsers] = useState([]);

  // 📌 ตรวจสอบ Token ก่อนโหลด ManageUser
  useEffect(() => {
    const token = localStorage.getItem("authToken");
    if (!token) {
      navigate("/"); // ถ้าไม่มี Token ให้กลับไปที่หน้า Login
    }
  }, [navigate]);

  // 📌 ดึงข้อมูลผู้ใช้
  useEffect(() => {
    axios
      .get("http://localhost:3000/api/admin/users", {
        headers: { Authorization: `Bearer ${localStorage.getItem("authToken")}` },
      })
      .then((res) => setUsers(res.data))
      .catch((err) => console.error("Error fetching users:", err));
  }, []);

  // 📌 เปลี่ยนสถานะผู้ใช้
  const updateUserStatus = (userID, status) => {
    axios
      .put(
        `http://localhost:3000/api/admin/users/${userID}/status`,
        { status },
        { headers: { Authorization: `Bearer ${localStorage.getItem("authToken")}` } }
      )
      .then(() => {
        setUsers(users.map((user) => (user.UserID === userID ? { ...user, Status: status } : user)));
      })
      .catch((err) => console.error("Error updating status:", err));
  };

  // 📌 ลบผู้ใช้ (Soft Delete)
  const deleteUser = (userID) => {
    if (window.confirm("ต้องการลบผู้ใช้นี้ใช่หรือไม่?")) {
      axios
        .delete(`http://localhost:3000/api/admin/users/${userID}`, {
          headers: { Authorization: `Bearer ${localStorage.getItem("authToken")}` },
        })
        .then(() => {
          setUsers(users.filter((user) => user.UserID !== userID));
        })
        .catch((err) => console.error("Error deleting user:", err));
    }
  };

  // 📌 Logout Function
  const handleLogout = () => {
    localStorage.removeItem("authToken"); // ลบ Token
    navigate("/"); // Redirect ไปที่หน้า Login
  };

  return (
    <ManageUserContainer>
      <Header>
        <span>Manage Users</span>
        <ButtonGroup>
          <Button color="#3498db" hoverColor="#2980b9" onClick={() => navigate("/dashboard")}>
            Back to Dashboard
          </Button>
          <Button color="red" hoverColor="darkred" onClick={handleLogout}>
            Logout
          </Button>
        </ButtonGroup>
      </Header>
      <ContentContainer>
        <Table>
          <thead>
            <tr>
              <Th>User ID</Th>
              <Th>Email</Th>
              <Th>Username</Th>
              <Th>Role</Th>
              <Th>Status</Th>
              <Th>Actions</Th>
            </tr>
          </thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.UserID}>
                <Td>{user.UserID}</Td>
                <Td>{user.Email}</Td>
                <Td>{user.Username}</Td>
                <Td>{user.Role}</Td>
                <Td>{user.Status}</Td>
                <Td>
                  {user.Status === "active" ? (
                    <Button color="red" hoverColor="darkred" onClick={() => updateUserStatus(user.UserID, "suspended")}>
                      Suspend
                    </Button>
                  ) : (
                    <Button color="green" hoverColor="darkgreen" onClick={() => updateUserStatus(user.UserID, "active")}>
                      Activate
                    </Button>
                  )}
                  <Button color="gray" hoverColor="black" onClick={() => deleteUser(user.UserID)}>
                    Delete
                  </Button>
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </ContentContainer>
    </ManageUserContainer>
  );
}

export default ManageUser;
