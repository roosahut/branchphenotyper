import axios from "axios";
import { UploadResponse } from "../components/Predictions";

export const uploadImages = async (images: File[]) => {
  try {
    const formData = new FormData();
    images.forEach(image => formData.append("images", image));

    const response = await axios.post<UploadResponse[]>(`/api/upload`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data
  } catch (error) {
    console.error("Error uploading image:", error);
    throw error;
  }
};