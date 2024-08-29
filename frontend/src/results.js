import React from "react";
function Results({ response, scrollToResults }) {
  const threshold = 0.5;
  const isPneumonia = response && Number(response.model_prediction) >= threshold;
  
  
    return (
      <div ref={scrollToResults} className="flex h-screen bg-gray-300 flex-col  items-center">
        <h1 className="m-5">Results</h1>
        <h2 className="m-5 text-2xl">Model Prediction percentage</h2>
        {response ? (
        <div className="flex items-center flex-col bg-white p-6 rounded-lg w-fit shadow-md pb-28">
          <p className="text-lg mb-2">Result: <span className="font-semibold">{response.prediction}</span></p>
          <p className="text-lg mb-2">Model prediction: <span className="font-semibold">{response.model_prediction}</span></p>
          {isPneumonia ? (
            <div>
              <p className="text-lg text-red-600 font-semibold">The model predicts a high likelihood of pneumonia.</p>
              
            </div>
          ) :(
            <p className="text-lg text-blue-600 font-semibold">The model predicts a low likelihood of pneumonia</p>
          )}
          <div className="mt-5 p-4 bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700">
                <p className="font-semibold">Disclaimer:</p>
                <p>The results provided by the model may be inaccurate. Please consult a medical professional for a definitive diagnosis.</p>
          </div>
        </div>
        
      ) : (
        <p className="text-lg text-gray-700">No results yet!</p>
      )}
      
      </div>
    );
  }
  export default Results;
  