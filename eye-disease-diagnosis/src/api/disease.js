import axios from 'axios';

// 设置后端地址和端口号
const API_BASE_URL = 'http://localhost:3000';

// 单人疾病预测
export const predictDisease = async (formData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/disease_prediction`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response;
  } catch (error) {
    console.error('Error predicting disease:', error);
    throw error;
  }
};

// 批量疾病预测
export const batchPredictDisease = async (formData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/batch_disease_prediction`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response;
  } catch (error) {
    console.error('Error batch predicting disease:', error);
    throw error;
  }
};