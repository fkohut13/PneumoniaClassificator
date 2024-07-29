import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import SineWaves from 'sine-waves';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

reportWebVitals();

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Ensure SineWaves runs after the DOM is fully loaded and React has rendered
window.addEventListener('load', function() {
  var waves = new SineWaves({
    el: document.getElementById('waves'),
    
    speed: 4,
    
    width: function() {
      return window.innerWidth;
    },
    
    height: 200,
    
    ease: 'SineInOut',
    
    wavesWidth: '70%',
    
    waves: [
      {
        timeModifier: 3,
        lineWidth: 1,
        amplitude: 70,
        wavelength: 25
      },
      {
        timeModifier: 2,
        lineWidth: 1,
        amplitude: -70,
        wavelength: 25
      },
      {
        timeModifier: 1,
        lineWidth: 1,
        amplitude: -70,
        wavelength: 25
      },
   
    ],
  
    // Called on window resize
    resizeEvent: function() {
      var gradient = this.ctx.createLinearGradient(0, 0, this.width, 0);
      gradient.addColorStop(0,"rgba(0, 180, 216, 0.7)");
      gradient.addColorStop(0.5,"rgba(0, 119, 182, 0.7)");
      gradient.addColorStop(1,"rgba(3, 4, 94, 0.7)");
     
      var index = -1;
      var length = this.waves.length;
      while(++index < length){
        this.waves[index].strokeStyle = gradient;
      }
     
      // Clean Up
      index = void 0;
      length = void 0;
      gradient = void 0;
    }
  });
});
