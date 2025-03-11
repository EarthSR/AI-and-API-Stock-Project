import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/Login"; // หน้าที่เราได้สร้างขึ้น
import Dashboard from "./pages/Dashboard"; // หน้า Dashboard ที่ต้องการไปหลังจาก login สำเร็จ
import ManageUser from "./pages/ManageUser";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/manageuser" element={<ManageUser/>} />
      </Routes>
    </Router>
  );
}

export default App;
