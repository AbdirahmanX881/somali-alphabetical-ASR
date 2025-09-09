import React, { useEffect, useState } from 'react';
import { View, Text, FlatList } from 'react-native';
import api from '../../services/api';

export default function AlertListScreen() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/reports/all/');
        setAlerts(res.data);
      } catch (err) {
        console.warn(err.message);
      }
    })();
  }, []);

  return (
    <View style={{ flex: 1, padding: 12 }}>
      <FlatList data={alerts} keyExtractor={(i) => String(i.id)} renderItem={({ item }) => (
        <View style={{ padding: 12, borderBottomWidth: 1 }}>
          <Text style={{ fontWeight: '600' }}>{item.type}</Text>
          <Text>{item.description}</Text>
        </View>
      )} />
    </View>
  );
}