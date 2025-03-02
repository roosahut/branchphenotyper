import React from "react";
import "./PredictionDisplay.css";

export interface Prediction {
  weeping: number;
  antigravitropic: number;
  main_trunks: number;
  canopy_breadth: number;
  primary_branches: number;
  branch_density: number;
  orientation: number;
}

interface Props {
  filename: string;
  predictions: Prediction;
}

const PredictionDisplay: React.FC<Props> = ({ filename, predictions }) => {
  return (
    <div className="prediction-card">
      <h3>{filename}</h3>
      <ul>
        <li>Weeping: {predictions.weeping}</li>
        <li>Antigravitropic: {predictions.antigravitropic}</li>
        <li>Main Trunks: {predictions.main_trunks}</li>
        <li>Canopy Breadth: {predictions.canopy_breadth}</li>
        <li>Primary Branches: {predictions.primary_branches}</li>
        <li>Branch Density: {predictions.branch_density}</li>
        <li>Orientation: {predictions.orientation}</li>
      </ul>
    </div>
  );
};

export default PredictionDisplay;
