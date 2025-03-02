import React from "react";
import "./Predictions.css";
import PredictionDisplay, { Prediction } from "./PredictionDisplay";

export interface UploadResponse {
  filename: string;
  predictions?: Prediction;
  error?: string;
}

interface Props {
  uploadResponses: UploadResponse[];
}

const Predictions: React.FC<Props> = ({ uploadResponses }) => {
  if (uploadResponses.length === 0) return null;

  const downloadCSV = () => {
    if (uploadResponses.length === 0) return;

    const csvRows = [
      [
        "Filename",
        "Weeping (0-5)",
        "Antigravitropic (0-1)",
        "Main Trunks",
        "Canopy Breadth",
        "Primary Branches (0-5)",
        "Branch Density (0-5)",
        "Orientation (0-1)",
      ],
      ...uploadResponses.map((response) => {
        if (!response.predictions) return [];
        return [
          response.filename,
          response.predictions.weeping,
          response.predictions.antigravitropic,
          response.predictions.main_trunks,
          response.predictions.canopy_breadth,
          response.predictions.primary_branches,
          response.predictions.branch_density,
          response.predictions.orientation,
        ].join(",");
      }),
    ];

    const csvContent =
      "data:text/csv;charset=utf-8," +
      csvRows
        .map((row) => (Array.isArray(row) ? row.join(",") : row))
        .join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    const fileName = `Branch_Phenotypes_${
      new Date().toISOString().split("T")[0]
    }.csv`;

    link.setAttribute("href", encodedUri);
    link.setAttribute("download", fileName);
    document.body.appendChild(link);
    link.click();
  };

  return (
    <div className="predictions-container">
      <h2>Predictions</h2>
      <div className="prediction-list">
        {uploadResponses.map((response, index) =>
          response.predictions ? (
            <div key={index} className="prediction-item">
              <PredictionDisplay
                filename={response.filename}
                predictions={response.predictions}
              />
            </div>
          ) : (
            <p key={index} className="error-message">
              {response.filename}: {response.error}
            </p>
          )
        )}
      </div>
      <button onClick={downloadCSV} className="csv-button">
        Download CSV
      </button>
    </div>
  );
};

export default Predictions;
