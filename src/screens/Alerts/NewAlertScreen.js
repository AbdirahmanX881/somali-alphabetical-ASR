import React, { useState } from 'react';
import { View, TextInput, Button, Alert } from 'react-native';
import api from '../../services/api';

export default function NewAlertScreen({ navigation }) {
  const [description, setDescription] = useState('');
  const [type, setType] = useState('accident');

  const submit = async () => {
    try {
      // Add current GPS coords or map-picked coords
      const payload = { type, description, latitude: 2.046934, longitude: 45.318161 };
      await api.post('/reports/', payload);
      Alert.alert('Sent', 'Alert submitted');
      navigation.goBack();
    } catch (err) {
      Alert.alert('Error', err.response?.data?.detail || err.message);
    }
  };

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <TextInput placeholder="Short description" value={description} onChangeText={setDescription} style={{ borderWidth: 1, padding: 8, marginBottom: 12 }} />
      <Button title="Submit" onPress={submit} />
    </View>
  );
}
