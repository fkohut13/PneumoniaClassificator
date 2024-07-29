import { useState } from "react";
import arrowdown from "./arrowdown.svg";
import darkmode from "./darkmode.svg";

function Header({ scrollToRef }) {
  const handleScroll = () => {
    if (scrollToRef.current) {
      scrollToRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };
  const [lightmode, setDarkmode] = useState()
  function changewebstyle() {
    if(!setDarkmode) {
      

    }
  }

  return (
    <header className="App-header">
      <button onClick={changewebstyle} className="darkmode-btn">
        <img src={darkmode} alt="Dark Mode" />
      </button>
      <ul>
        <li>
          <h1 className="welcome">Welcome</h1>
        </li>
        <li>
          <h2>Pneumonia Classificator</h2>
        </li>
      </ul>
      <div className="waves-container">
        <canvas id="waves"></canvas>
      </div>
      <div className="arrow-container">
        <p>Get started!</p>
        <button onClick={handleScroll} className="arrowdown">
          <img src={arrowdown} alt="Arrow Down" />
        </button>
      </div>
    </header>
  );
}

export default Header;
