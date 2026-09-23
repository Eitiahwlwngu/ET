const API_BASE_URL = 'http://localhost:3000';

export const getUserInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/getuserinfo`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching user info:', error);
    return null;
  }
};

export const updateUserInfo = async (userData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/upuserinfo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error updating user info:', error);
    return null;
  }
};

export const changePassword = async (passwordData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/changepwd`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(passwordData),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error changing password:', error);
    return null;
  }
};