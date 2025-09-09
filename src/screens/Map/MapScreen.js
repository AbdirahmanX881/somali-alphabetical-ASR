import React, { useEffect, useState, useRef } from 'react';
import { View, Text, Button } from 'react-native';
import MapView, { Marker, PROVIDER_GOOGLE } from 'react-native-maps';
import api from '../../services/api';
import { useNavigation } from '@react-navigation/native';

const MOGADISHU_REGION = {
  latitude: 2.046934,
  longitude: 45.318161,
  latitudeDelta: 0.1,
  longitudeDelta: 0.1,
};

export default function MapScreen() {
  const [markers, setMarkers] = useState([]);
  const mapRef = useRef(null);
  const navigation = useNavigation();

  useEffect(() => {
    fetchNearbyAlerts();
    // TODO: subscribe to WebSocket for live updates
  }, []);

  const fetchNearbyAlerts = async () => {
    try {
      // Example call - adapt endpoint & params to backend
      const res = await api.get('/reports/nearby/', { params: { lat: MOGADISHU_REGION.latitude, lng: MOGADISHU_REGION.longitude, radius_km: 25 } });
      setMarkers(res.data || []);
    } catch (err) {
      console.warn('Failed to fetch alerts', err.message);
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <MapView
        ref={mapRef}
        provider={PROVIDER_GOOGLE}
        style={{ flex: 1 }}
        initialRegion={MOGADISHU_REGION}
      >
        <Marker coordinate={{ latitude: MOGADISHU_REGION.latitude, longitude: MOGADISHU_REGION.longitude }} title="Mogadishu" description="City center" />
        {markers.map((m) => (
          <Marker key={m.id} coordinate={{ latitude: m.latitude, longitude: m.longitude }} title={m.type} description={m.description} />
        ))}
      </MapView>

      <View style={{ position: 'absolute', top: 40, left: 12 }}>
        <Button title="New Alert" onPress={() => navigation.navigate('NewAlert')} />
      </View>
    </View>
  );
}
