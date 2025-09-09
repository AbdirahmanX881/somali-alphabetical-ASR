import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:8000/api/'; // for Android emulator; change to your local IP or server

const instance = axios.create({ baseURL: BASE_URL, timeout: 10000 });

const api = {
  get: (url, config) => instance.get(url, config),
  post: (url, data, config) => instance.post(url, data, config),
  put: (url, data, config) => instance.put(url, data, config),
  delete: (url, config) => instance.delete(url, config),
  setToken: (token) => {
    if (token) instance.defaults.headers.common.Authorization = `Bearer ${token}`;
    else delete instance.defaults.headers.common.Authorization;
  }
};

export default api;
