import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000';

// // 用户登录API
// export const login = (userData) => {
//   return axios.post(`${API_BASE_URL}/userlogin`, {
//     userData
//   });
// };

// 用户登录API
export const login = (userData) => {
  return axios.post(`${API_BASE_URL}/login`, userData, {
    headers: {
      'Content-Type': 'application/json' // 确保发送的请求是 JSON 格式
    }
  });
};

// 用户注册API
export const register = (userData) => {
  return axios.post(`${API_BASE_URL}/register`, userData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded' // 确保发送的请求是表单格式
    }
  });
};

// // 用户注册API
// export const register = (userData) => {
//   return axios.post(`${API_BASE_URL}/userreg`, {
//     userData
//   });
// };

// // 检查是否登录API
// export const checkIsLoginApi = () => {
//   return axios.post(`${API_BASE_URL}/islogin`);
// };

// // 用户退出登录API
// export const userExitApi = () => {
//   return axios.post(`${API_BASE_URL}/userexit`);
// };

// // 上传用户头像API
// export const upUserHeadApi = (file) => {
//   const formData = new FormData();
//   formData.append('file', file);
//   return axios.post(`${API_BASE_URL}/upuserhead`, formData);
// };

// // 获取用户信息API
// export const getUserInfoApi = () => {
//   return axios.post(`${API_BASE_URL}/getuserinfo`);
// };

// // 更新用户个人信息API
// export const upUserInfoApi = (headpic, nickname, tel, email, introduce) => {
//   return axios.post(`${API_BASE_URL}/upuserinfo`, {
//     headpic,
//     nickname,
//     tel,
//     email,
//     introduce
//   });
// };




// 修改用户密码API
export const changePwd = (currentPwd, newPwd, confirmNewPwd) => {
  return axios.post(`${API_BASE_URL}/changepwd`, {
    current_pwd: currentPwd,
    new_pwd: newPwd,
    confirm_new_pwd: confirmNewPwd
  });
};

// 获取用户信息API
export const getUserPageInfor = (uid) => {
  return axios.get(`${API_BASE_URL}/getuserpageinfor`, {
    params: {
      uid
    }
  });
};

//更新用户个人信息API
export const upUserInfo = (headpic,nickname,tel,email,introduce)=>{
  return axios.get(`${API_BASE_URL}/upuserinfoApi`,{
    headpic,
    nickname,
    tel,
    email,
    introduce
    });
};