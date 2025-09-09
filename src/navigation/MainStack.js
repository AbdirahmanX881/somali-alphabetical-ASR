import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import MapScreen from '../screens/Map/MapScreen';
import AlertListScreen from '../screens/Alerts/AlertListScreen';
import NewAlertScreen from '../screens/Alerts/NewAlertScreen';
import ProfileScreen from '../screens/Profile/ProfileScreen';

const Stack = createNativeStackNavigator();

export default function MainStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Map" component={MapScreen} />
      <Stack.Screen name="AlertList" component={AlertListScreen} />
      <Stack.Screen name="NewAlert" component={NewAlertScreen} />
      <Stack.Screen name="Profile" component={ProfileScreen} />
    </Stack.Navigator>
  );
}