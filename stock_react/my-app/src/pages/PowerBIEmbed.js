import React from "react";
import styled from "styled-components";

const PowerBIContainer = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
`;

const StyledIframe = styled.iframe`
  width: 100%;
  height: 100%;
  border: none; /* เอาขอบ iframe ออก */
`;

const PowerBIEmbed = ({ embedUrl }) => {
  return (
    <PowerBIContainer>
      <StyledIframe id="powerBIFrame" title="Power BI Report" src={embedUrl} allowFullScreen />
    </PowerBIContainer>
  );
};

export default PowerBIEmbed;
