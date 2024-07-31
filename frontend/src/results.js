import React from "react";
function Results() {
    const pneumoniaPercentage = 10; 
  
    return (
      <div className="flex h-screen bg-gray-300 flex-col">
        <h1 className="m-5">Results</h1>
        <h2 className="m-5 text-2xl">Model Prediction percentage</h2>
        <div className="flex items-center flex-col">         
          <p className="m-2">Pneumonia Detected</p>
          <div className="w-5/6 bg-gray-200 rounded-full h-8 dark:bg-gray-700 relative">
            <div
              className="bg-red-400 h-8 rounded-full"
              style={{ width: `${pneumoniaPercentage}%` }}
            >
              <span className="absolute left-1/2 transform -translate-x-1/2 text-white font-bold">
                {pneumoniaPercentage}%
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }
  export default Results;
  