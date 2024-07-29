function Prediction({ scrollToRef }) {
  return (
    <div ref={scrollToRef} className="about-container flex justify-evenly">
      <div className="h-screen flex items-center justify-around p-3">
        <div className="w-full max-w-md p-9 bg-white rounded-lg shadow-lg">
          <h1 className="text-center text-2xl sm:text-2xl font-semibold mb-4 text-gray-800">Your X-ray image here!</h1>
          <div className="bg-gray-100 p-8 text-center rounded-lg border-dashed border-2 border-gray-300 hover:border-blue-500 transition duration-300 ease-in-out transform hover:scale-105 hover:shadow-md" id="dropzone">
            <label for="fileInput" className="cursor-pointer flex flex-col items-center space-y-2">
              <svg className="w-16 h-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
              </svg>
              <span className="text-gray-600">Drag and drop your files here</span>
              <span className="text-gray-500 text-sm">(or click to select)</span>
            </label>
            <input type="file" id="fileInput" className="hidden" multiple />
          </div>
          <div className="mt-6 text-center" id="fileList"></div>
        </div>
      </div>

      <div className="h-screen flex items-center justify-around p-3">
        <div className="w-full max-w-md p-9 bg-white rounded-lg shadow-lg">
          <div> <h1 className="text-center text-2xl sm:text-2xl font-semibold mb-4 text-gray-800">Prediction</h1></div>
          <div><img/></div>
        </div>
        
      </div>
    </div>
  );
}

export default Prediction;
