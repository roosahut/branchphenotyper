import React from "react";
import ImageUploader from "./components/ImageUploader";
import "./App.css";

const App: React.FC = () => {
  return (
    <div className="app-container">
      <h1 className="title">Branchphenotyper</h1>
      <p className="subtitle">
        Upload an image of a birch to analyze tree branches!
      </p>
      <ImageUploader />
    </div>
  );
};

export default App;
