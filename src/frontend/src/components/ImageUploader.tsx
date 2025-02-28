import React, { useState } from "react";
import { uploadImages } from "../services/routes";
import "./ImageUploader.css";

const ImageUploader: React.FC = () => {
  const [images, setImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadResponses, setUploadResponses] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleImageChange = (files: FileList) => {
    const fileArray = Array.from(files);
    setImages((prev) => [...prev, ...fileArray]);
    setUploadResponses([]);
    setErrorMessage(null);
    setPreviews((prev) => [
      ...prev,
      ...fileArray.map((file) => URL.createObjectURL(file)),
    ]);
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
      setUploadResponses(
        responses.map(
          (response) => `${response.message} (File: ${response.filename})`
        )
      );
      setImages([]);
      setPreviews([]);
    } catch (error) {
      setErrorMessage("Failed to upload the image - " + error);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => prev.filter((_, i) => i !== index));
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
        <p>or</p>
        <label htmlFor="file-upload" className="file-upload-label">
          Choose File
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="file-input"
        />
      </div>
      {previews.length > 0 && (
        <div className="image-preview-container">
          {previews.map((preview, index) => (
            <div key={index} className="image-preview">
              <img src={preview} alt={`Preview ${index}`} className="preview" />
              <button
                onClick={() => handleDelete(index)}
                className="delete-button"
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}

      {images && (
        <button
          onClick={handleUpload}
          className="upload-button"
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload images"}
        </button>
      )}
      {uploadResponses.length > 0 && (
        <div className="success-message">
          {uploadResponses.map((response, index) => (
            <p key={index}>{response}</p>
          ))}
        </div>
      )}
      {errorMessage && <p className="error-message">{errorMessage}</p>}
    </div>
  );
};

export default ImageUploader;
