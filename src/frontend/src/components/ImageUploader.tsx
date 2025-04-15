import React, { useState } from "react";
import { uploadImages } from "../services/routes";
import "./ImageUploader.css";
import Predictions, { UploadResponse } from "./Predictions";

const ImageUploader: React.FC = () => {
  const [images, setImages] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadResponses, setUploadResponses] = useState<UploadResponse[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleImageChange = (files: FileList) => {
    const fileArray = Array.from(files);
    setImages((prev) => [...prev, ...fileArray]);
    setUploadResponses([]);
    setErrorMessage(null);
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      handleImageChange(event.target.files);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    if (event.dataTransfer.files) {
      handleImageChange(event.dataTransfer.files);
    }
  };

  const handleUpload = async () => {
    if (images.length === 0) return;
    setUploading(true);
    try {
      const responses = await uploadImages(images);
      setUploadResponses(responses);
      setImages([]);
    } catch (error) {
      setErrorMessage("Failed to upload the images - " + error);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="upload-container">
      <div
        className={`drop-zone ${dragActive ? "drag-active" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p>Drag & Drop an image here</p>
        <p>(You can select multiple images)</p>
        <p>or</p>
        <label htmlFor="file-upload" className="file-upload-label">
          Choose File
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileInput}
          className="file-input"
        />
      </div>
      {images.length > 0 && (
        <div className="image-list-container">
          <ul className="image-list">
            {images.map((image, index) => (
              <li key={index} className="image-list-item">
                {image.name}
                <button
                  onClick={() => handleDelete(index)}
                  className="delete-button"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {images.length > 0 && (
        <button
          onClick={handleUpload}
          className="upload-button"
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload images"}
        </button>
      )}
      {uploadResponses.length > 0 && (
        <Predictions uploadResponses={uploadResponses} />
      )}
      {errorMessage && <p className="error-message">{errorMessage}</p>}
    </div>
  );
};

export default ImageUploader;
