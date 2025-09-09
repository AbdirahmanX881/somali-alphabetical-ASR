import React, { createContext, useContext, useEffect, useState } from 'react';
import * as SecureStore from 'expo-secure-store';
import api from '../services/api';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const token = await SecureStore.getItemAsync('authToken');
      if (token) {
        api.setToken(token);
        // fetch profile
        try {
          const res = await api.get('/users/me/');
          setUser(res.data);
        } catch (err) {
          console.warn('Failed to fetch user', err.message);
          await SecureStore.deleteItemAsync('authToken');
        }
      }
      setLoading(false);
    })();
  }, []);

  const signIn = async (credentials) => {
    const res = await api.post('/auth/login/', credentials);
    const token = res.data.access || res.data.token;
    await SecureStore.setItemAsync('authToken', token);
    api.setToken(token);
    const me = await api.get('/users/me/');
    setUser(me.data);
    return me.data;
  };

  const signOut = async () => {
    await SecureStore.deleteItemAsync('authToken');
    api.setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);