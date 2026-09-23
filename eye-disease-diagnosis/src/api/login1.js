
const API_BASE_URL = 'http://localhost:3000';

export const register = async (formData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error in register API:', error);
    return { res: 'error', message: error.message };
  }
};

export const login = async (formData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error in login API:', error);
    return { res: 'error', message: error.message };
  }
};

export const checkUsername = async (username) => {
  try {
    const response = await fetch(`${API_BASE_URL}/check-username?username=${username}`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error in checkUsername API:', error);
    return { exists: false };
  }
};

export const checkTel = async (tel) => {
  try {
    const response = await fetch(`${API_BASE_URL}/check-tel?tel=${tel}`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error in checkTel API:', error);
    return { exists: false };
  }
};

export const getVerificationCode = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/verification-code`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error in getVerificationCode API:', error);
    return null;
  }
};