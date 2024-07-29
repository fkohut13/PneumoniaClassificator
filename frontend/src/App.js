import React, { useRef, useState } from 'react';
import Header from './header';
import About from './about';
import "./App.css";

function App() {
  const scrollToRef = useRef(null);
  const changewebstyle = useState(null);

  return (
    <div className="App">
      <Header scrollToRef={scrollToRef} />
      <body>
      <About scrollToRef={scrollToRef} />
      </body>
    </div>
  );
}

export default App;
