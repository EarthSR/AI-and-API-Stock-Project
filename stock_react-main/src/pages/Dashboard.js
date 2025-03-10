import React from "react";
import PowerBIEmbed from "./PowerBIEmbed";
import styled from "styled-components";

const DashboardContainer = styled.div`
  display: flex;
  height: 100vh;
  background: #121212;
  overflow: hidden;
`;

const Sidebar = styled.div`
  width: 250px;
  background: #1e1e1e;
  padding: 20px;
  color: white;
  font-weight: bold;
  box-shadow: 4px 0 10px rgba(0, 0, 0, 0.3);
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  overflow: hidden;
`;

const Header = styled.header`
  width: 100%;
  background: #ff8c00;
  padding: 10px;
  text-align: center;
  color: white;
  font-size: 24px;
  font-weight: bold;
  box-shadow: 0 4px 8px rgba(255, 140, 0, 0.4);
`;

function Dashboard() {
  return (
    <DashboardContainer>
      <Sidebar>
        <h2>📊 Power BI Dashboard</h2>
        <div>Overview</div>
        <div>Stock Analysis</div>
        <div>Market Trends</div>
        <div>Reports</div>
      </Sidebar>
      <MainContent>
        <Header>Stock Market Performance</Header>
        <PowerBIEmbed />
      </MainContent>
    </DashboardContainer>
  );
}

export default Dashboard;
