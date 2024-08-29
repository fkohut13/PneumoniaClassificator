import 'boxicons';
import React, { useState } from 'react';
import Alert from '@mui/material/Alert';
import { Button } from '@mui/material';

const Prediction = ({ scrollToRef, setResponse, scrollToResults}) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [filename, setFilename] = useState(null);
  const [xrayPredicted, setxrayPredicted] = useState(null);
  const [uploadcomplete, setUploadcomplete] = useState(false);
  const [showNoFileAlert, setShowNoFileAlert] = useState(false);
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);
  const [showErrorAlert, setShowErrorAlert] = useState(false);
  const [showseeResultsAlert, setshowseeResultsAlert] = useState(false)
  const [showScrollResultBTN, setShowScrollResultBTN] = useState(false)
  const [tryagain, settryagain] = useState(false);

  function getFilename(file) {
    return file.name;
  }

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setShowNoFileAlert(false);
    if (selectedFile) {
      const fileName = getFilename(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setFilename(fileName);
    }
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
    setShowNoFileAlert(false);
    if (droppedFile) {
      const fileName = getFilename(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setFilename(fileName);
    }
  };

  const handleRemove = () => {
    setFile(null);
    setPreview(null);
    setFilename(null);
    setUploadcomplete(false);
  };

  const handleTryagain = () => {
    setFile(null)
    setPreview(null);
    setFilename(null);
    setUploadcomplete(false);
    setxrayPredicted(null);
    settryagain(false);
    setShowScrollResultBTN(false);
  }

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUpload = async () => {
    if (!file) {
      setShowNoFileAlert(true);
      setTimeout(() => setShowNoFileAlert(false), 3000);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setShowSuccessAlert(true);
      setUploadcomplete(true);
      setTimeout(() => setShowSuccessAlert(false), 3000); 
    } catch (error) {
      setShowErrorAlert(true); 
      setTimeout(() => setShowErrorAlert(false), 3000);
    }
  };

  const handlePrediction = async () => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
      });
      const data = await response.json();
      setResponse(data)
      handleGetXray(); // Fetch the predicted X-ray image after prediction
      settryagain(true);
      setshowseeResultsAlert(true)
      setShowScrollResultBTN(true)
      setTimeout(() => setshowseeResultsAlert(false), 3000);
    } catch (error) {
      setShowErrorAlert(true);
    }
  };

  const handleScroll = () => {
    if (scrollToResults.current) {
      scrollToResults.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleGetXray = async () => {
    try {
      const response = await fetch('http://localhost:5000/get-xray', {
        method: 'GET',
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setxrayPredicted(imageUrl);
        console.log('Image fetched successfully');
      } else {
        console.error('Failed to fetch image');
        setShowErrorAlert(true);
      }
    } catch (error) {
      console.error('Error fetching image:', error);
      setShowErrorAlert(true);
    }
  };

  return (
    <div ref={scrollToRef} className="about-container flex justify-center">
      {showNoFileAlert && (
        <Alert severity="error" className="fixed top-0 left-0 right-0 m-4">
          Upload a file first!
        </Alert>
      )}
      {showSuccessAlert && (
        <Alert severity="success" className="fixed top-0 left-0 right-0 m-4">
          File uploaded successfully!
        </Alert>
      )}
      {showErrorAlert && (
        <Alert severity="error" className="fixed top-0 left-0 right-0 m-4">
          Error uploading file!
        </Alert>
      )}
      {showseeResultsAlert && (
        <Alert severity="success" className="fixed top-0 left-0 right-0 m-4">
          See Results below!
        </Alert>
      )}
      
      <div className="h-screen flex items-center justify-around p-3">
        {!tryagain && (
           <div className="w-full max-w-md p-9 bg-white rounded-lg shadow-lg">
           {!file && (
             <h1 className="text-center text-2xl sm:text-2xl font-semibold mb-4 text-gray-800">
               Your X-ray image here!
             </h1>
           )}
           {!file && (
             <div
               className="bg-gray-100 p-8 text-center rounded-lg border-dashed border-2 border-gray-300 hover:border-blue-500 h-60 w-60 transition duration-300 ease-in-out transform hover:scale-105 hover:shadow-md"
               id="dropzone"
               onDrop={handleFileDrop}
               onDragOver={handleDragOver}
             >
               <label htmlFor="fileInput" className="cursor-pointer flex flex-col items-center space-y-2">
                 <svg className="w-16 h-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                 </svg>
                 <span className="text-gray-600">Drag and drop your files here</span>
                 <span className="text-gray-500 text-sm">(or click to select)</span>
               </label>
               <input type="file" id="fileInput" className="hidden" onChange={handleFileChange} />
             </div>
           )}
           {file && (
             <div className="text-center">
               <p className="text-center mb-4 text-gray-800">
                 <b>File name: </b>{filename}
               </p>
               <img src={preview} className="h-60 w-60 rounded-lg" alt="X-ray preview" />
               {!uploadcomplete && (
                 <div className='m-2 flex justify-between'>
                   <Button onClick={handleRemove} variant="outlined" size='small' startIcon={<box-icon name='trash-alt' type='solid' color='#8CBAE8' ></box-icon>}>
                     Remove
                   </Button>
                   <Button onClick={handleUpload} variant="contained" size='small' endIcon={<box-icon name='cloud-upload' type='solid' ></box-icon>}>
                     Upload
                   </Button>
                 </div>
               )}
             </div>
           )}
         </div>
        )}
      </div>

      <div className="h-screen flex items-center justify-around p-3 flex-col">
        <div className="w-full max-w-md p-4 bg-white rounded-lg shadow-lg">
          <div>
            <h1 className="text-center text-2xl sm:text-2xl font-semibold mb-4 text-gray-800">Prediction</h1>
          </div>
          <div className="bg-gray-100 text-center rounded-lg border-spacing-3 h-auto max-w-full border-2 border-gray-300 hover:border-blue-500 transition duration-300 ease-in-out">
            {xrayPredicted ? (
              <img src={xrayPredicted} className="h-full w-full object-cover rounded-lg" alt="X-ray Predicted" />
            ) : (
              <div className=' p-8 h- w-60'>
                <p className="text-gray-600">No prediction available</p>
                <box-icon name='bot' color='#8c8787' ></box-icon>
                
              </div>
              
              
              
            )}
          </div>
          {uploadcomplete && (
            <div className='mt-1 flex justify-evenly'>
              <Button onClick={handlePrediction} variant="contained" size='small' endIcon={<box-icon name='scan' color='white'></box-icon>}>
                Predict
              </Button>

              {tryagain && (
                 <Button onClick={handleTryagain} variant="outlined" size='small' >
                 Try Again!
               </Button>
              )}
             
            </div>
          )}
         
        </div>
        {showScrollResultBTN && (
            <div className="flex flex-col items-center justify-center ">
            <p className="text-black">Results</p>
            <button onClick={handleScroll} className="  flex justify-center items-center bg-slate-300 p-1 w-14 h-11 rounded-2xl hover:scale-125 transition-all ease-in-out ">
            <box-icon name='chevrons-down' color="blue"></box-icon>
            </button>
          </div>
          )}
        
      </div>
      
      
    </div>
    
  );
};

export default Prediction;
