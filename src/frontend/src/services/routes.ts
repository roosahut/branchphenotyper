import axios from "axios";

export interface UploadResponse {
  filename: string;
  predictions?: {
            weeping: number,
            antigravitropic: number,
            main_trunks: number,
            canopy_breadth: number,
            primary_branches: number,
            branch_density: number,
            orientation: number
        }
  error?: string;
}

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