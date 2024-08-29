import { useState } from "react";
import { IconButton } from "@mui/material";
function Header({ scrollToRef }) {
  const handleScroll = () => {
    if (scrollToRef.current) {
      scrollToRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="App-header">
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
      <div className="flex flex-col items-center justify-center ">
        <p className="text-black">Get started!</p>
        <button onClick={handleScroll} className="  flex justify-center items-center bg-slate-300 p-1 w-14 h-11 rounded-2xl hover:scale-125 transition-all ease-in-out ">
        <box-icon name='chevrons-down' color="blue"></box-icon>
        </button>
      </div>
    </header>
  );
}

export default Header;
