import React from 'react';
import { View, Text, Button } from 'react-native';
import { useAuth } from '../../context/AuthContext';

export default function ProfileScreen() {
  const { user, signOut } = useAuth();
  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, marginBottom: 8 }}>My Profile</Text>
      <Text style={{ marginBottom: 4 }}>{user?.email}</Text>
      <Button title="Sign out" onPress={signOut} />
    </View>
  );
}