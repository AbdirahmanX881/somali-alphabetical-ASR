import React, { useState } from 'react';
import { View, TextInput, Button, Alert, Text } from 'react-native';
import api from '../../services/api';

export default function RegisterScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const onRegister = async () => {
    try {
      await api.post('/auth/register/', { email, password });
      Alert.alert('Registered', 'Please login');
      navigation.navigate('Login');
    } catch (err) {
      Alert.alert('Registration error', err.response?.data?.detail || err.message);
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', padding: 16 }}>
      <Text style={{ fontSize: 20, marginBottom: 10 }}>Create account</Text>
      <TextInput placeholder="Email" value={email} onChangeText={setEmail} style={{ marginBottom: 8, borderWidth: 1, padding: 8 }} />
      <TextInput placeholder="Password" value={password} onChangeText={setPassword} secureTextEntry style={{ marginBottom: 12, borderWidth: 1, padding: 8 }} />
      <Button title="Register" onPress={onRegister} />
    </View>
  );
}
