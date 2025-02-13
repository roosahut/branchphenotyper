import React, { useState } from "react";
import { uploadImage } from "../services/routes";
import "./ImageUploader.css";

const ImageUploader: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleImageChange = (file: File) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      handleImageChange(event.target.files[0]);
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
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      handleImageChange(event.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setUploading(true);
    try {
      const response = await uploadImage(image);
      alert("Image uploaded successfully!");
      console.log("Response:", response);
    } catch (error) {
      alert("The connection with backend is not yet built - " + error);
    } finally {
      setUploading(false);
    }
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
      {preview && <img src={preview} alt="Preview" className="preview" />}
      {image && (
        <button
          onClick={handleUpload}
          className="upload-button"
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload Image"}
        </button>
      )}
    </div>
  );
};

export default ImageUploader;
