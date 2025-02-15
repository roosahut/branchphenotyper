import axios from "axios";

export const uploadImage = async (image: File) => {
  try {
    const formData = new FormData();
    formData.append("file", image);

    const response = await axios.post(`/api/upload`, formData, {
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