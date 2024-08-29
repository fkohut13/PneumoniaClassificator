import React, { useRef, useState } from 'react';
import Header from './header';
import Prediction from './prediction';
import Results from './results';
import Footer from './footer';
import "./App.css";

function App() {
  const scrollToRef = useRef(null);
  const scrollToResults = useRef(null);
  const  [response, setResponse] = useState(null)

  return (
    <div className="App">
      <Header scrollToRef={scrollToRef} />
      <body>
      <Prediction scrollToRef={scrollToRef} scrollToResults={scrollToResults} setResponse={setResponse} />
      <Results scrollToResults={scrollToResults} response={response}/>
      </body>
      <Footer/>
    </div>
  );
}
 
export default App;